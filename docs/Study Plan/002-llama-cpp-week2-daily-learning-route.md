# llama.cpp 第 2 周每日学习路线

本路线根据 [llama.cpp 8 周深度学习与二次开发计划](llama-cpp-8-week-learning-plan.md) 的“第 2 周: Transformer 结构巩固、GGUF、模型加载和设备放置”拆分而来。第 1 周解决的是一次推理中 token 如何经过 decoder block 变成下一个 token；本周解决的是这个 decoder block 所需的超参数和权重从哪里来，以及它们如何到达可执行的 backend buffer。

本周要贯通的主线如下：

```text
GGUF 文件
  -> header / metadata / tensor info / tensor data
  -> llama_model_loader
  -> architecture -> llama_model subclass
  -> hparams + vocab + named weight tensors
  -> CPU/GPU buffer type selection
  -> mmap or read/upload
  -> llama_model owned weight buffers
```

完成本路线后，应能从一个真实模型的 `n_embd`、`n_head`、`n_head_kv`、`n_layer`、`n_vocab` 还原 Attention、FFN 和 KV cache 的主要 shape；能从 `llama_model_load_from_file()` 口述到 tensor buffer；并能用一次受控实验解释 `mmap`、`mlock`、普通读取和 layer offload 的差别。

## 使用约定

- 工作日每天 2 小时，周末每天 4 小时，总计约 18 小时。每天先用 10 分钟闭卷回忆前一天的 3 个事实，结束前用 15 分钟记录“已确认事实、证据位置、仍待验证的问题”。
- 本周不进入 `build_arch_graph()`、GGML kernel 或 sampler 实现。这些是第 3 周和后续周的内容。看到 tensor 时先记录“语义、GGML `ne[]`、逻辑 shape、所属 layer、所在 buffer”，不要顺着 op 深入。
- 以下命令默认在仓库根目录执行。总计划的默认模型是 `Llama-3.2-3B-Instruct-f16.gguf`；若它不在本地，可使用当前仓库已有的较小 GGUF，但笔记必须如实记录模型名和 `general.architecture`，不要把另一种架构的 metadata 当成 Llama 的结果。

  ```bash
  cd /home/king/llama.cpp
  export MODEL="$PWD/Llama-3.2-3B-Instruct-f16.gguf"
  test -f "$MODEL" || export MODEL="$PWD/deepseek-r1-1.5b-f16.gguf"
  test -f "$MODEL"
  git rev-parse --short HEAD
  ```

- `llama-gguf` 的当前用法是 `llama-gguf data.gguf r|w [n]`。本周使用 `r n` 读取元数据和 tensor info，`n` 表示不校验完整 tensor data，避免把文件检查误认为模型推理。
- `llama-bench` 的当前 load mode 为 `--load-mode mmap|none|mlock|mmap+mlock|dio`。旧的 `-mmp` 已标记为 deprecated，不作为本周主命令。CPU 对照是硬性验收；只有 backend 和驱动都可用时才进行 GPU placement 观察。
- 学习笔记、日志和脚本放在仓外的 `llama-study/`，不修改 `src/`、`ggml/` 或模型文件。建议新增：

  ```text
  llama-study/
    notes/week2-day{1..7}.md
    results/week2/day{1..7}/
    experiments/week2-load-observe.sh
  ```

## 本周追踪表

每次打开源文件或运行命令，都把新事实记入同一张表。表中“逻辑 shape”使用 Transformer 习惯的 `[out, in]` 或 `[V, D]` 描述；“GGML `ne[]`”以实际输出为准，不能擅自假设两个维度的顺序相同。

| 项目 | 本周要记录的内容 | 证据 |
| --- | --- | --- |
| 模型身份 | 文件名、commit、`general.architecture`、量化类型、split 数 | `llama-gguf` 输出和运行日志 |
| 核心 hparams | `n_ctx_train`、`n_embd`、`n_layer`、`n_head`、`n_head_kv`、`n_ff`、`n_vocab` | GGUF key 和 `load_hparams()` |
| 一个 block 的权重 | `blk.0` 中 norm、Q/K/V、output、FFN gate/up/down 的 name/type/shape | GGUF tensor info 和 `load_arch_tensors()` |
| 文件到内存 | metadata、tensor info、tensor data 各自何时读取；是否 `mmap` | loader 断点、`init_mappings()`、日志 |
| 放置 | tensor 的目标 buffer type、实际 model buffer，CPU/GPU layer 数 | `load_tensors()`、`select_weight_buft()`、启动日志 |
| 排除项 | compute buffer 和 KV buffer 的用途与 model weight buffer 的区别 | context 创建日志；本周不追踪其内部实现 |

## Day 1: 从 GGUF 输出建立真实模型档案

**投入: 2 小时。** 今天先观察文件格式和模型配置，不跟踪 loader 的 C++ 调用。

### 学习路线

1. **0-10 分钟，闭卷回忆**：从第 1 周的单层图写出 `D`、`H`、`KvH`、`Dh`、`I`、`V` 的含义，以及 `Dh = D / H` 在普通 Llama attention 中成立的前提。
2. **10-30 分钟，区分 GGUF 三类数据**：阅读 [`ggml/src/gguf.cpp`](../../ggml/src/gguf.cpp)，只定位 `gguf_context`、`gguf_kv`、`gguf_tensor_info` 的定义。画出 `header -> kv metadata -> tensor info -> aligned tensor data` 四段图，并写清 tensor info 包含 name、type、dimensions、data offset，而不等于 tensor data 本身。
3. **30-65 分钟，读取模型**：运行：

   ```bash
   ./build-study-debug/bin/llama-gguf "$MODEL" r n
   ```

   将完整输出保存为 `llama-study/results/week2/day1/llama-gguf.txt`。从输出填写模型身份和 7 个核心 hparams；同时摘录 `general.architecture`、`general.quantization_version`、`split.count`（若存在）、tokenizer 相关 key 的证据行。
4. **65-100 分钟，建立 shape 表**：对于 Llama 架构，按 `D=n_embd`、`H=n_head`、`KvH=n_head_kv`、`Dh=D/H`、`I=n_ff`、`V=n_vocab` 写出下表中的逻辑 shape。实际模型不是 Llama 时，保留表结构并标出不能直接成立的假设。

   | 权重语义 | 常见 GGUF 名称 | Llama 逻辑 shape |
   | --- | --- | --- |
   | token embedding | `token_embd.weight` | `[V, D]` |
   | Q projection | `blk.0.attn_q.weight` | `[H * Dh, D]` |
   | K projection | `blk.0.attn_k.weight` | `[KvH * Dh, D]` |
   | V projection | `blk.0.attn_v.weight` | `[KvH * Dh, D]` |
   | attention output | `blk.0.attn_output.weight` | `[D, D]` |
   | FFN gate/up | `blk.0.ffn_gate.weight`, `blk.0.ffn_up.weight` | `[I, D]` |
   | FFN down | `blk.0.ffn_down.weight` | `[D, I]` |
   | output projection | `output.weight` 或 tied embedding | `[V, D]` |

5. **100-120 分钟，检查数量级**：任选 Q、K、FFN gate 三个权重，用 shape 计算元素数量；解释 `KvH < H` 时 K/V 投影为什么比 Q 投影小。列出“逻辑 shape”和 GGML `ne[]` 仍未确认的区别，留给 Day 5 验证。

### 当天输出与验收

- `llama-study/notes/week2-day1.md`：GGUF 四段图、真实模型档案、hparams 表、上述 weight shape 表和三个元素数量计算。
- `llama-study/results/week2/day1/llama-gguf.txt`：原始工具输出。
- 验收：能说清 metadata、tensor info、tensor data 的边界；能从真实 `n_embd/n_head/n_head_kv/n_ff/n_vocab` 写出 Q、K、V、FFN 的主要逻辑 shape；能解释 metadata 是描述模型而不是模型权重本身。
- 未通过时：重新检查 `n_head_kv` 与 `n_head`，再用 `KvH = H` 和 `KvH < H` 各写一次 K weight 和 KV cache 的 head 维度，不进入 Day 2。

## Day 2: 从超参数回到 Attention、FFN 和 KV cache

**投入: 2 小时。** 今天把 Day 1 的文件字段连接回第 1 周的 Transformer 心智模型。

### 学习路线

1. **0-10 分钟**：不看 Day 1 笔记，写出模型的 `D/H/KvH/I/V/L` 实际数值或标出尚未找到的项。
2. **10-40 分钟，读通用 hparams**：阅读 [`src/llama-model.cpp`](../../src/llama-model.cpp) 的 `llama_model_base::load_hparams()`，定位通用的 context length、embedding length、block count、metadata copy 和 `rope_type` 设置。阅读 [`src/models/llama.cpp`](../../src/models/llama.cpp) 的 `llama_model_llama::load_arch_hparams()`，区分“所有模型共有的 hparams”与“Llama 架构特有的 hparams”。
3. **40-75 分钟，做一次配置推导**：以真实数值补全以下数据流，不要求创建任何 tensor：

   ```text
   hidden [N, D]
     -> Q [N, H, Dh]
     -> K/V [N, KvH, Dh]
     -> score [H, N, S]
     -> attention output [N, D]
     -> SwiGLU intermediate [N, I]
     -> block output [N, D]
     -> logits [N, V]
   ```

   `N` 是本次 decode 输入 token 数，`S` 是包含历史 KV 的可见长度。写明 `N` 与训练 context length、运行时 `n_ctx` 都不是同一概念。
4. **75-105 分钟，估算 KV cache**：只按 K/V 的元素数估算单层、单 sequence 的最小 cache 数据量：

   ```text
   elements per layer = 2 * S * KvH * Dh
   bytes per model   = 2 * S * KvH * Dh * L * bytes_per_element
   ```

   分别用 `S=128` 和计划使用的 `n_ctx` 计算。记录 cache type 的假设，例如 F16 是 2 bytes；不要把该估算误写成实际 `llama_memory_i` 分配大小。
5. **105-120 分钟，口述**：用“模型文件提供的数字”解释 GQA 如何减少 K/V 权重和 cache，而不会减少 Q head 数。

### 当天输出与验收

- `llama-study/notes/week2-day2.md`：真实配置到 Transformer shape 的映射表、一次完整 shape 推导、两组 KV cache 元素/字节估算。
- 验收：能区分 `n_ctx_train`、运行时 `n_ctx`、本次 batch token 数 `N` 和历史长度 `S`；能从 hparams 解释 Q 与 K/V 维度不同的原因；能说明 KV cache 不是 GGUF 中已保存的权重。

## Day 3: 从公共加载 API 到 GGUF loader

**投入: 2 小时。** 今天跟踪路径选择、GGUF metadata 的初始化和 split 文件处理，但不进入 tensor placement。

### 学习路线

1. **0-10 分钟**：闭卷写出 Day 2 的 Q/K/V shape 和 cache 估算公式。
2. **10-35 分钟，找公共入口**：阅读 [`src/llama.cpp`](../../src/llama.cpp) 的 `llama_model_load_from_file_impl()`、`llama_model_load()` 和 `llama_model_load_from_file()`。画出公共 API 至 `llama_model_load()` 的调用图，标出 `llama_model_params` 在哪里传入。
3. **35-75 分钟，读 loader 构造**：阅读 [`src/llama-model-loader.cpp`](../../src/llama-model-loader.cpp) 的 `llama_model_loader::llama_model_loader()`。为下面每步补一行“输入、结果、尚未读取的内容”：

   ```text
   path_model
     -> gguf_init_from_file(... no_alloc = true)
     -> general.architecture -> llm_arch
     -> llama_file + GGML metadata context
     -> tensor metadata -> weights_map
     -> split.count / split.no validation -> extra files and contexts
   ```

   特别写明 `no_alloc = true` 在这里避免的是为 tensor data 分配/读取，不是让 metadata 消失。
4. **75-100 分钟，处理 split 分支**：对照 `llama_get_list_splits()` 与构造函数的 `n_split > 1` 分支，画出单文件与多 split 的差异。手写一个假想文件名 `model-00001-of-00004.gguf`，说明为什么加载必须从第一个 split 开始，以及 `weights_map` 为什么不允许重复 tensor name。
5. **100-120 分钟，运行观察**：对当前单文件模型运行：

   ```bash
   LLAMA_TRACE=1 ./build-study-debug/bin/llama-bench \
     -m "$MODEL" -ngl 0 -p 1 -n 0 -r 1 --no-warmup --load-mode mmap
   ```

   保存 stderr 中的 loader/model buffer 日志。即使单文件没有额外 split 日志，也要把“该分支未触发”的证据写进笔记。

### 当天输出与验收

- `llama-study/notes/week2-day3.md`：公共 API 到 loader 的调用图、单文件/多 split 对照图、`no_alloc` 的解释。
- `llama-study/results/week2/day3/load-mmap.log`：一次加载日志和完整命令。
- 验收：能解释 loader 先读 GGUF metadata 而不是立即把全部权重读入 RAM；能解释 `general.architecture` 如何参与后续模型选择；能说明 split 文件的验证目标。

## Day 4: 模型工厂、hparams 和词表加载顺序

**投入: 2 小时。** 今天跟踪 metadata 如何变成正确的 C++ 模型对象。

### 学习路线

1. **0-10 分钟**：闭卷画出 Day 3 的 `path_model -> loader -> arch` 链路。
2. **10-40 分钟，读模型工厂**：阅读 [`src/llama-model.cpp`](../../src/llama-model.cpp) 中的 `llama_model_create(llama_model_loader &, ...)`、`llama_model_create(llm_arch, ...)` 和 `llama_model_mapping()`。只跟踪当前模型对应的 `LLM_ARCH_*` 分支；不要逐个阅读其他模型类。
3. **40-75 分钟，读固定加载顺序**：回到 `llama_model_load()`，为下面步骤写明依赖关系与失败时的含义：

   ```text
   llama_model_loader
     -> llama_model_create
     -> llama_prepare_model_devices
     -> load_hparams
     -> load_vocab
     -> load_stats
     -> load_tensors
   ```

   解释为什么架构、hparams 和 vocab 必须在 `load_tensors()` 前可用；说明 `vocab_only` 会停止在哪一步，以及它不等于 tokenizer 不需要 metadata。
4. **75-105 分钟，定位多态边界**：阅读 [`src/llama-model.h`](../../src/llama-model.h) 中 `llama_model` 和 `llama_model_base` 的虚函数声明。画出一张小表，列出 `load_hparams()` 的通用部分、`load_arch_hparams()` 的架构变化点、`load_vocab()`、`load_arch_tensors()` 与 `build_arch_graph()` 的职责。最后一项只标边界，不读实现。
5. **105-120 分钟，GDB 准备**：运行第 1 周的 `llama-simple` 命令进入 GDB，设置 pending breakpoint：

   ```gdb
   set pagination off
   set breakpoint pending on
   break llama_model_load_from_file_impl
   break llama_model_load
   break llama_model_create
   break llama_model_base::load_hparams
   break llama_model_base::load_tensors
   run
   ```

   每次命中记录函数、当前架构名、是否已经有 model object、`n_gpu_layers` 和是否已进入 tensor loading。若符号没有按短名匹配，使用 `info functions llama_model` 或 `rbreak llama_model_.*load.*` 后重新设置，不修改源码。

### 当天输出与验收

- `llama-study/notes/week2-day4.md`：模型工厂图、加载顺序表、通用逻辑与架构逻辑的边界。
- `llama-study/results/week2/day4/gdb-factory.txt`：断点记录或无法解析符号时的诊断记录。
- 验收：能说清 `general.architecture` 到 C++ 子类的过程；能说明 `load_hparams/load_vocab/load_tensors` 的先后依赖；能解释本周只到权重准备完成，尚未进入 graph execution。

## Day 5: tensor 名称、GGML 维度和 buffer type 选择

**投入: 2 小时。** 今天把 Day 1 的逻辑 shape 落到真实 tensor metadata 与设备放置决策。

### 学习路线

1. **0-10 分钟**：闭卷列出 `load_tensors()` 前必须已完成的三件事。
2. **10-45 分钟，读 Llama tensor 声明**：阅读 [`src/models/llama.cpp`](../../src/models/llama.cpp) 的 `load_arch_tensors()`，只覆盖 token embedding、output norm/output、layer 0 的 norm、Q/K/V/output 和 FFN gate/up/down。将 `create_tensor(..., { ... })` 的 `ne[]` 顺序逐项抄入 Day 1 表，再在相邻列写逻辑 shape。不要仅凭矩阵数学转置它；以源码和 GGUF 输出为准。
3. **45-75 分钟，追一个 tensor**：在 `llama-gguf` 输出中选择 `blk.0.attn_q.weight`；若当前架构名称不同，选择语义等价的第一层 query weight，并在笔记中说明差异。完成下表：

   | 字段 | 记录内容 |
   | --- | --- |
   | GGUF name | 完整名称 |
   | metadata type | 例如 F16 或某种量化类型 |
   | GGML `ne[]` | 按工具输出原样记录 |
   | Transformer 语义 | Q projection，输入/输出维度 |
   | loader mapping | `weights_map` 中的 file index 与 data offset |
   | declared tensor | `create_tensor()` 对应行 |
   | target buffer type | Day 5 后半段确认 |

4. **75-105 分钟，读 placement 决策**：阅读 [`src/llama-model-loader.cpp`](../../src/llama-model-loader.cpp) 的 `select_weight_buft()` 和 `get_tensor()` 附近的选择逻辑，再阅读 [`src/llama-model.cpp`](../../src/llama-model.cpp) 中 `llama_model_base::load_tensors()` 的 CPU/GPU buffer type list、layer split 和 model buffer 日志。写出决策顺序：显式 tensor override、可用 buffer type、优先 device list、CPU fallback；不要把 `n_gpu_layers` 解释成“所有内存都移到 GPU”。
5. **105-120 分钟，读日志分类**：从 Day 3 日志抄出一行 `model buffer size`，再运行一次最小推理或 bench，抄出 compute/KV buffer 相关日志。为三类 buffer 各写一句“何时创建、主要存什么、由谁拥有”。

### 当天输出与验收

- `llama-study/notes/week2-day5.md`：至少一个 tensor 的完整追踪表、GGML `ne[]` 与逻辑 shape 对照、buffer selection 决策图、三类 buffer 对比。
- 验收：能解释 tensor name 不是随意字符串，而是架构代码声明与 GGUF 文件匹配的键；能指出 `model buffer` 存权重、`compute buffer` 服务执行时临时张量、KV buffer 保存运行时状态；能解释无可用 GPU 时 `-ngl 0` 的 CPU placement。

## Day 6: mmap、普通读取和 placement 的受控实验

**投入: 4 小时。** 今天用可重复测量验证“mmap 不等于已经将全部数据读入物理内存”，并把加载参数与日志对应起来。

### 学习路线

1. **0-25 分钟，实验设计**：创建 `llama-study/results/week2/day6/`。固定模型、commit、线程数、`-ngl 0`、prompt tokens、generation tokens 和 repetition；只改变 `--load-mode`。写下假设：`mmap` 建立文件映射，实际页访问可能延后；`none` 要为权重分配 buffer 并显式读取。不要预先断言哪个模式一定更快或 RSS 一定更小。
2. **25-85 分钟，CPU 基线**：各运行一次或按机器状态重复两次：

   ```bash
   /usr/bin/time -v ./build-study-debug/bin/llama-bench \
     -m "$MODEL" -ngl 0 -p 32 -n 0 -r 1 --no-warmup \
     --load-mode mmap

   /usr/bin/time -v ./build-study-debug/bin/llama-bench \
     -m "$MODEL" -ngl 0 -p 32 -n 0 -r 1 --no-warmup \
     --load-mode none
   ```

   分别保存 stdout、stderr 和 `/usr/bin/time -v` 的 Maximum resident set size。若命令因为 Debug build 过慢，先保持相同 Debug build 完成一次正确性对照，再单独记录 Release build 性能数据，绝不混在同一张性能表中。
3. **85-125 分钟，解释数据**：列出 wall time、最大 RSS、模型 buffer 大小、pp 结果和异常信息。写至少两个影响因素：文件已在 page cache、首次访问 page fault、模型量化/大小、compute/KV buffer、Debug/Release。结论必须仅限于本机/本模型/本参数，不能由一轮结果推广为通用性能定律。
4. **125-175 分钟，GDB 观察 data path**：用一个较短 benchmark 或 `llama-simple` 进入 GDB，在以下位置设置断点：

   ```gdb
   set breakpoint pending on
   break llama_model_loader::init_mappings
   break llama_model_loader::load_data_for
   break llama_model_loader::load_all_data
   run
   ```

   分别在 `mmap` 与 `none` 模式记录：`use_mmap`、`cur->data` 来自 mapping 地址还是已分配 buffer、是否经过 `read_raw()`。若断点次数太多，先在 `load_all_data()` 停住并只对第一层 Q tensor使用条件断点或 source line，记录方法本身。
5. **175-220 分钟，可选 GPU placement**：仅在 `llama-bench --list-devices` 显示可工作的非 CPU device 且运行不报 driver/runtime 错误时，固定 load mode，比较 `-ngl 0` 与较小的正值，例如 `-ngl 1`。保存“offloading ... layers”和各 model buffer 行。若环境不可用，记录原因并停止，不安装驱动、不改 backend build。
6. **220-240 分钟，整理**：完成实验结果表和结论。明确本周没有用性能数据评价 GPU kernel，只验证权重 placement 与加载路径。

### 当天输出与验收

- `llama-study/results/week2/day6/mmap.*`、`none.*`：完整命令、stdout、stderr、time 输出；可选 GPU 对照日志。
- `llama-study/notes/week2-day6.md`：实验设计、结果表、mmap data path 与普通读取的证据、局限性。
- 验收：能用源码和日志分别解释 `mmap` 和普通读取的路径；能说明 RSS 不是“模型已经全部被计算”的证据；能从日志辨别 `-ngl` 改变的是哪些权重 placement，而非 KV cache 语义。

## Day 7: 闭卷加载链重建与第二周验收

**投入: 4 小时。** 今天不读新模块，只验证能否将文件格式、模型抽象、tensor 和设备 buffer 串成一条完整链路。

### 学习路线

1. **0-35 分钟，闭卷绘图**：从 `llama_model_load_from_file()` 开始，画到一个 `blk.0.attn_q.weight` 被放入 model buffer。图中必须包含 `gguf_init_from_file(no_alloc=true)`、architecture、model factory、`load_hparams`、`load_vocab`、`load_arch_tensors`、`init_mappings`、buffer allocation 或 mapping、`load_all_data`。
2. **35-80 分钟，闭卷回答**：不看源码回答本周 5 个必答问题：

   1. `n_embd`、`n_head`、`n_head_kv` 如何决定 Q/K/V 和 KV cache 的主要 shape？
   2. GGUF metadata、tensor info、tensor data 分别何时被读取，分别用来做什么？
   3. `general.architecture` 如何决定 `llama_model` 子类？
   4. `mmap` 为什么不等于所有权重立即进入物理 RAM？
   5. `n_gpu_layers`、buffer type、model buffer、compute buffer 和 KV buffer 各自负责什么？

3. **80-135 分钟，空白终端验证**：不翻 Day 1-6 的命令，重新完成一次 `llama-gguf "$MODEL" r n`、一次 CPU `llama-bench --load-mode mmap -ngl 0`，并从日志找出 model buffer 行。命令或参数记不住可以查 `--help`，但不要复制旧命令。
4. **135-185 分钟，源码定位验收**：在 15 分钟内定位并解释以下位置的职责：`llama_model_load_from_file_impl()`、`llama_model_loader` 构造函数、`llama_model_create()`、`llama_model_base::load_hparams()`、Llama `load_arch_tensors()`、`init_mappings()`、`load_all_data()`。每项写一个输入和一个输出。
5. **185-220 分钟，证据一致性检查**：交叉核对 Day 1 的 shape 表、Day 5 的真实 tensor 追踪、Day 6 的日志和实验结论。修正任何把 GGML `ne[]`、Transformer 逻辑 shape、训练 context、运行时 context、model/compute/KV buffer 混为一谈的记录，并保留修正原因。
6. **220-240 分钟，周复盘**：填写下方清单，列出进入第 3 周前仍要问的 3 个问题。问题应指向 graph builder，例如“已加载的 `ggml_tensor *` 如何被 Attention op 引用”，不要提前阅读答案。

### 第 2 周验收清单

- [ ] 已保存一次 `llama-gguf "$MODEL" r n` 原始输出，并能从中确认模型架构、核心 hparams、tensor 类型和至少一个 layer 0 tensor。
- [ ] 能用真实 `n_embd/n_head/n_head_kv/n_layer/n_vocab/n_ff` 推出 Q、K、V、FFN、logits 和 KV cache 的主要 shape。
- [ ] 能区分 GGUF metadata、tensor info 和 tensor data，且能解释 `no_alloc = true` 的作用范围。
- [ ] 能从 `llama_model_load_from_file()` 讲到 loader、architecture factory、hparams/vocab 和 `load_tensors()`。
- [ ] 已完成一个 `blk.0` 权重从 GGUF name/type/shape/offset 到 `create_tensor()` 和目标 buffer type 的追踪表。
- [ ] 能解释 `mmap`、`mlock`、普通读取的概念边界，并完成至少 `mmap` 与 `none` 的 CPU 对照或记录了明确阻塞原因。
- [ ] 能区分 model weight buffer、compute buffer、KV buffer，且不把 `n_gpu_layers` 当成完整运行时内存策略。
- [ ] 已保留 GDB 或等价日志证据，说明加载阶段先处理 metadata/hparams，后处理 tensor data。

## 本周完成包

进入第 3 周前，`llama-study/` 中至少应有：

```text
notes/week2-day1.md       # GGUF 档案和 shape 表
notes/week2-day2.md       # hparams 到 Transformer/KV cache 推导
notes/week2-day3.md       # API 到 loader、split 对照
notes/week2-day4.md       # 工厂和加载顺序
notes/week2-day5.md       # tensor 追踪和 buffer type
notes/week2-day6.md       # mmap/none 实验与 GDB 观察
notes/week2-day7.md       # 闭卷验收和周复盘
results/week2/day1/       # llama-gguf 原始输出
results/week2/day3/       # CPU load 日志
results/week2/day4/       # factory GDB 记录
results/week2/day6/       # mmap/none 对照及可选 GPU 日志
```

若模型文件不可用，Day 1 的格式阅读、Day 3-5 的源码追踪与 Day 7 的闭卷图可以先完成；Day 1 的真实档案、Day 3 的加载日志、Day 6 的对照实验和 Day 7 的命令验收必须在取得本地 GGUF 后补做。CUDA driver/runtime 不匹配时，只记录 GPU 观察未完成的原因，CPU 验收仍是进入第 3 周的门槛。
