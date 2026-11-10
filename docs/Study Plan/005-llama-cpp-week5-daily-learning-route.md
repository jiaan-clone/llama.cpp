# llama.cpp 第 5 周每日学习路线: KV Cache、序列管理、上下文移动和状态保存恢复

本路线根据 [llama.cpp 8 周深度学习与二次开发计划](llama-cpp-8-week-learning-plan.md) 的第 5 周安排展开, 承接 [第 4 周: llama_context、batch、ubatch、encode/decode 主调用链](004-llama-cpp-week4-daily-learning-route.md)。本周沿着 `memory->init_batch()` 深入, 回答历史状态放在哪里、多个 sequence 怎样共享它、何时可以回收, 以及恢复状态后怎样正确继续生成。

本周主线:

```text
llama_context owns llama_memory_i
  -> ordinary Transformer: llama_kv_cache
       -> cells: position / sequence membership / pending shift
       -> per-layer K/V tensors: actual cached values
       -> init_batch -> prepare/find_slot -> memory context
       -> apply -> graph inputs -> compute writes K/V
  -> sequence operations: rm / cp / keep / add / div
       -> metadata changes
       -> pending copy/shift -> memory update when needed
  -> state: serialize memory -> clear/load -> decode pending input -> sample
  -> recurrent/hybrid: same interface, different state and constraints
```

完成标准: 能画出共享前缀和分叉后的 cell/sequence 关系, 解释 context shift 的位置更新与 K 更新, 用 token ID 对照证明恢复后续写一致, 并说明普通 KV、recurrent 和 hybrid memory 的共同接口与不同限制。

## 使用约定与本周边界

- 工作日每天 2 小时, 周末每天 4 小时, 共 18 小时。每天开头 10 分钟闭卷回忆, 结尾至少 15 分钟整理证据和未解决的问题。
- 本文按工作区 `545ddb30a` 核对。总计划基准版本较早, 函数、参数和序列化内容以当前 checkout 为准; 阅读时用符号定位, 不依赖固定行号。
- 必做路径采用 CPU、普通 decoder-only Transformer、F16 K/V、关闭 Flash Attention、无 LoRA、无 backend sampling。共享前缀实验显式设置 `kv_unified=true`, 固定合法的 sequence 数量和短输入。
- 沿用本地 `deepseek-r1-1.5b-f16.gguf`, 实际架构为 Qwen2。实验记录写清模型架构, 不把结果标为 Llama 权重实测。可替换为完整 Llama GGUF, 但前后对照必须使用同一模型和运行参数。
- 独立实验改造第 1 周的最小推理程序, 放在 `/home/king/llama-study/experiments/`, 不要求修改核心源码。阅读和实验遵守 [AGENTS.md](../../AGENTS.md) 与 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
- 本周固定使用简单 greedy sampler 验证自建实验; sampler chain 和随机性留到第 6 周。recurrent/hybrid 以源码对照为必做项, 无需额外下载模型。
- 本文给出学习任务和实验设计, 并不表示这些任务已经完成。现有测试能证明的范围与自建实验应补充的断言分别列出。

开始前闭卷回答: `llama_decode()` 如何得到当前 ubatch? 什么时候应用 memory metadata, 什么时候真正执行图? 为什么复制 KV 后仍不能直接采样出另一个 sequence 的下一 token? 回答不清时先回看第 4 周的调用链和输出索引。

## 7 天安排

| 天 | 时间 | 核心问题 | 当天主要输出 |
| --- | ---: | --- | --- |
| Day 1 | 2h | 为什么 memory 同时参与 batch 准备和图输入? | 接口职责图、更新与计算边界 |
| Day 2 | 2h | token、position、cell、slot 和 stream 如何对应? | cell 映射、slot 分配推演 |
| Day 3 | 2h | 共享、删除、保留、位置移动分别改变什么? | 五类操作前后图、context shift 推演 |
| Day 4 | 2h | state 包含什么, 恢复后从哪一步继续? | API 对照、测试断言表、checkpoint 约定 |
| Day 5 | 2h | recurrent/hybrid 如何复用接口而改变语义? | 三类 memory 对照、实验预期表 |
| Day 6 | 4h | 共享前缀后两个分支是否隔离, shift 是否按预期执行? | 独立实验、greedy 对照、GDB 证据 |
| Day 7 | 4h | 保存/清空/恢复能否保持续写一致? | 恢复实验、已有测试、parallel 日志、周复盘 |

## 统一术语与记录表

| 名称 | 本周含义 | 需要避免的混淆 |
| --- | --- | --- |
| token ID | 词表中的编号 | 同一个 token ID 可以出现在很多 position |
| position | 模型输入的逻辑位置, 普通模型用于 RoPE 等 | 不等于 KV buffer 的物理下标 |
| sequence ID | memory 中的序列归属 | 不天然等于线程 ID、请求 ID 或客户端 ID |
| cell | 对某个物理槽位的 metadata, 包括位置和 sequence 集合 | 不是一份完整的应用会话, 也不是单层 K/V 对象 |
| K/V tensors | 各层实际缓存的数据 | 与 cell metadata 分开存储 |
| slot_info | 当前 ubatch 分配到的 stream 和 cell 下标 | 不等于一段必然连续的逻辑 token 区间 |
| stream | KV 实现中的存储分区 | `kv_unified=true` 时多个 sequence 可处于同一 stream |
| used cells | 当前占用的物理槽位数 | 共享后不等于各 sequence 长度之和 |
| `n_kv` | attention 使用的 KV 范围, 受布局和 padding 影响 | 不总等于 used cells |
| state bytes | 序列化出的 memory 等数据 | 不等于模型权重文件或完整应用进程快照 |

每次实验至少记录以下字段。`pos_min/max` 可用公共 API 查询, cell 归属和 pending shift 用 GDB 或 debug 日志观察; 只看位置极值不足以证明共享关系或占用数量。

| 阶段/操作 | seq ID | 输入 token 数和 pos | stream/cell 范围 | used cells | seq pos min/max | pending copy/shift | 返回值/字节数 | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 示例: 共享前缀后 | 0, 1 | 无新输入 | 同一组前缀 cells | 应保持不变 | 两序列相同 | 同 stream 无数据复制 | `seq_cp` 无返回值 | 断点和 metadata |

## Day 1: 通用 memory 接口与 decode 的接合点

**投入: 2 小时。** 先明确对象职责, 再进入具体存储实现。

### 学习路线

1. **0-10 分钟**: 闭卷画出第 4 周的 `decode -> init_batch -> process_ubatch -> graph_compute`。
2. **10-40 分钟**: 阅读 [`src/llama-memory.h`](../../src/llama-memory.h) 和 [`src/llama-memory.cpp`](../../src/llama-memory.cpp), 区分长期存在的 `llama_memory_i` 与服务于一次 batch/update 的 `llama_memory_context_i`。
3. **40-70 分钟**: 在 [`src/llama-context.cpp`](../../src/llama-context.cpp) 跟踪 `memory_update()`、`decode()` 和 `process_ubatch()`。给下表每行补一个调用者与返回结果的处理位置。

   | 接口 | 阅读时要回答的问题 |
   | --- | --- |
   | `init_batch()` | 如何切分 ubatch 并确认这些输入能放入 memory? |
   | `init_full()` | 为什么图预留需要模拟满 memory, 它是否代表实际推理? |
   | `init_update()` | 哪些待执行的 copy/shift 需要额外处理? |
   | `get_ubatch()/next()` | 当前批次在哪里, 如何推进? |
   | `apply()` | 当前 memory context 怎样应用 metadata 或执行 update? |
   | `get_status()` | `SUCCESS/NO_UPDATE/FAILED_PREPARE/FAILED_COMPUTE` 如何传回上层? |

4. **70-100 分钟**: 对照 [`src/llama-kv-cache.h`](../../src/llama-kv-cache.h)、[`src/llama-graph.cpp`](../../src/llama-graph.cpp) 中 attention 与 memory input 的构建逻辑。找出 KV 读取、K/V 写入位置、attention mask 与 slot/sequence 信息的关系。
5. **100-120 分钟**: 画出两个时间点: metadata 已应用; graph 已完成并写入 K/V。解释为什么在前者断点看到的 occupied cell 不证明新 token 的 K/V 已计算完成。

### 当天输出与验收

- 接口职责图、普通 decode 与 pending update 两条路径。
- 能说明 model memory、GGML compute buffer、logits output buffer 分别保存什么。
- 能解释 memory 参与 ubatch 切分的原因: 可用槽位、sequence 布局和状态类型会约束本次计算输入。

## Day 2: KV cell、slot_info 和分配过程

**投入: 2 小时。** 用一个小例子追踪逻辑 token 如何落到物理槽位。

### 学习路线

1. **0-10 分钟**: 闭卷区分 token ID、position、sequence ID 与 cell index。
2. **10-40 分钟**: 阅读 [`src/llama-kv-cells.h`](../../src/llama-kv-cells.h) 的 `pos`、sequence bitset、`shift`、used 集合和 `seq_pos` 统计。再读 KV cache 的 `v_cells`、`v_heads`、`seq_to_stream` 和各层 K/V tensor。
3. **40-75 分钟**: 在 [`src/llama-kv-cache.cpp`](../../src/llama-kv-cache.cpp) 沿 `init_batch -> prepare -> find_slot -> apply_ubatch` 阅读。区分 `prepare()` 为验证容量所做的临时应用/还原, 与 `llama_kv_cache_context::apply()` 对当前 ubatch 的实际应用。`apply_ubatch()` 断点命中后先看 `bt`, 不能全部算成真实提交。
4. **75-100 分钟**: 手算下面的布局。cell 下标只作示例, 实际 allocator 可以选择其他可用位置。

   | cell | position | sequence 集合 | 解释 |
   | ---: | ---: | --- | --- |
   | 0 | 0 | `{0,1}` | 共享前缀 token 0 |
   | 1 | 1 | `{0,1}` | 共享前缀 token 1 |
   | 2 | 2 | `{0}` | 分支 A 的第一个私有 token |
   | 3 | 2 | `{1}` | 分支 B 的第一个私有 token |
   | 4 | 空 | `{}` | 可供后续输入使用 |

   回答: 两个 sequence 各有 3 个位置, 为什么只用了 4 个 cells? cell 2 与 cell 3 的 position 相同, 为什么不会互相参与另一分支的 attention? 删除 sequence 0 后哪些 cells 仍然存在?

5. **100-120 分钟**: 查看 `get_n_kv()` 和 `find_slot()` 的连续/非连续分配条件, 说明总空闲空间、连续空闲空间、attention 访问范围与预分配 buffer 大小是不同量。

### 一个容量手算

普通全层 attention、无特殊 cache 布局时, 用以下公式估算 K/V 数据主体, 再与启动日志核对:

```text
KV bytes ~= layers * total_physical_cells * (K_width * bytes_K + V_width * bytes_V)
K_width = KV_heads * key_head_dim
V_width = KV_heads * value_head_dim
```

本地 Qwen2 模型为 28 层、2 个 KV heads、head dimension 128。F16 K/V 下每个物理 cell 的跨层数据约为 `28 * (256 * 2 + 256 * 2) = 28672` bytes, 256 个 cells 约 7 MiB。该估算不含 metadata、对齐、compute buffer 和权重; `total_physical_cells` 要从实际配置确认, 不直接套用命令行 `-c` 解释所有 memory 布局。

### 当天输出与验收

- cell 映射图、slot 分配过程、容量估算与实际日志。
- 能解释前缀共享节省的是哪些重复状态, 以及为什么删除 sequence 不会使预分配 KV buffer 同比例缩小。
- 能在断点中区分 slot 规划、metadata 应用和 K/V tensor 写入。

## Day 3: sequence 操作与 context shift

**投入: 2 小时。** 今天先做纸面推演, Day 6 用独立程序核对。

### 学习路线

1. **0-10 分钟**: 重画 Day 2 的共享 cell 表。
2. **10-45 分钟**: 对照 [`include/llama.h`](../../include/llama.h) 的 memory API 与 KV 实现, 为每个操作画一组独立的前后状态。范围使用半开区间 `[p0,p1)`, 负 `p0/p1` 的全范围约定分别按接口确认。

   | 操作 | 普通 KV 主路径 | 必须记录的限制 |
   | --- | --- | --- |
   | `seq_cp(src,dst,p0,p1)` | 同 stream 给选中 cells 增加 dst 归属 | 本周 dst 为空; 跨 stream 涉及数据复制, 不能按零复制解释 |
   | `seq_rm(seq,p0,p1)` | 移除匹配的 sequence 归属, 无归属后 cell 才可复用 | 检查 bool 返回值, 其他 memory 的部分删除可能失败 |
   | `seq_keep(seq)` | 在本周 unified 布局保留目标序列, 去掉其他归属和无关 cells | 不把单 stream 实验机械推广成所有布局的清理行为 |
   | `seq_add(seq,p0,p1,delta)` | 修改 position 并记录位移 | 负移到无效位置可释放 cell; 共享 cell 的 position 不是每个 owner 独立保存 |
   | `seq_div(seq,p0,p1,d)` | 对 position 做整数除法, 记录对应位移 | `d > 1`, 可以产生重复 position; 不会自动减少 cell 数或扩容 |

3. **45-70 分钟**: 阅读 `seq_cp()` 的同 stream 与跨 stream 两个分支。后者当前要求完整 buffer 范围, 排队记录实际 buffer copy, 在后续 update 中执行。必做实验使用 unified 布局; 不尝试以跨 stream 的任意子区间复制代替前缀共享。
4. **70-100 分钟**: 阅读 [`tools/completion/completion.cpp`](../../tools/completion/completion.cpp) 的 `n_keep/n_discard` 分支, 再跟 `seq_add -> init_update -> update -> build_graph_shift`。按下例推演应用层 position 与 memory 的同步变化。
5. **100-120 分钟**: 解释 cache full 的处理边界: 返回码和日志提示容量问题; 应用选择结束、清理完成序列、移动上下文或重新 prefill, 不能假设 core 自动为任意输入提供无限上下文。

### context shift 的最小例子

只使用一个 sequence, 初始 position 为 `0..7`, 保留前 2 个 token, 丢弃中间 3 个 token:

```text
seq_rm (mem, 0, 2, 5)
seq_add(mem, 0, 5, 8, -3)

old positions: 0 1 [2 3 4] 5 6 7
new positions: 0 1         2 3 4
next input position: 5
```

- 占用从 8 个 cells 变成 5 个, 主要来自删除; 存活 cell 的逻辑 position 改变不等于其物理下标整体搬移。
- 先检查 `llama_memory_can_shift(mem)`。普通 RoPE KV 的待处理 shift 会触发 K 的旋转修正, V 不需要同样的 RoPE 修正; 不要把 `seq_add()` 返回当作相关 tensor 计算已完成。
- 应用的 `n_past/next_pos` 也要更新。只改 memory 而继续提交旧 position, 不能得到预期的连续上下文。
- 共享 cell 的 position 属于 cell 本身。不要对共享前缀中的一个 owner 单独 shift, 并假设另一个 owner 自动获得隔离副本; 必做 shift 实验使用单 sequence。
- shift 后保留的 K/V 曾经受到被删除历史的影响, 不要求它与“删掉文本后重新 prefill”的 logits 完全相同。

### 当天输出与验收

- 五类操作各一张前后状态图、shift 的 metadata/计算两阶段图。
- 能解释 `seq_rm()` 的归属删除、槽位复用与物理 buffer 释放是三个不同概念。
- 能说明 `can_shift=true` 的能力范围, 不把它解释为任意模型均支持任意历史回滚。

## Day 4: context state、sequence state 和恢复边界

**投入: 2 小时。** 重点是定义可恢复的计算位置, 以及逐条核对已有测试到底验证了什么。

### 学习路线

1. **0-10 分钟**: 闭卷列出继续生成所需的状态: model memory、下一个输入及其位置、输出、sampler、应用 token 历史。
2. **10-40 分钟**: 从 `include/llama.h` 的 state API 进入 `llama-context.cpp` 的 `state_write_data/state_read_data`、`state_seq_write_data/state_seq_read_data` 和文件包装。

   | API 组 | 当前源码中的主要内容/行为 | 本周检查项 |
   | --- | --- | --- |
   | `llama_state_get_size/get_data/set_data` | context 的架构标识与 memory 序列化 | 保存时获取大小, 检查写入/读取字节数 |
   | `llama_state_save_file/load_file` | 文件 magic/version、调用者提供的 token 列表、context state | 检查 bool、token 容量与实际加载数量 |
   | `llama_state_seq_get_size/get_data/set_data` | 单 sequence 的 memory 状态, 可恢复到指定目标 seq ID | 目标 ID 必须合法, 检查字节数与恢复后的位置 |
   | `llama_state_seq_save_file/load_file` | sequence state 的文件包装及 token 列表 | 不与完整 context 文件格式混用 |
   | `llama_memory_clear(mem,data)` | 清空 metadata, `data=true` 时也清零数据 buffer | 不释放 model, 不自动重置外部 sampler 和应用变量 |

3. **40-65 分钟**: 在 KV 的 `state_write/state_read` 中追踪 cell metadata 与各层 K/V tensor 的存取。说明为什么恢复后物理 cell 下标可以变化, 以及相同架构名不足以保证权重相同。
4. **65-100 分钟**: 阅读 [`tests/test-save-load-state.cpp`](../../tests/test-save-load-state.cpp) 与 [`common/common.cpp`](../../common/common.cpp) 的 `common_prompt_batch_decode()`。按下表填写每个测试的准备、动作和断言。
5. **100-120 分钟**: 写出下方 checkpoint 约定, 将“state 文件中有什么”和“应用还要保存什么”分成两列。

### 现有测试的证据范围

| 测试段 | 实际检查 | 不应据此推出的结论 |
| --- | --- | --- |
| Test 1 | 保存基线状态并生成参考 token 序列 | 此时已经完成所有恢复验证 |
| Test 2 | 删除 seq 0 前后, seq 1 的序列化字节保持一致 | 它使用 `seq_cp()` 验证了物理共享前缀; 实际上它分别 prefill 两个序列 |
| Test 3 | 新 context 加载文件, decode 最后一个 prompt token 后继续生成, token IDs 与参考一致 | load 本身恢复了可直接采样的 logits |
| Test 4 | host sequence state 保存、clear、迁移到另一 seq ID 后续写一致 | 任意新 context 只恢复 sequence state 就能直接 sample |
| Test 5 | 使用 `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` 路径迁移并续写 | 必须有 GPU 才能执行, 或该状态可作为通用磁盘文件 |

当前这些续写对照使用源码中显式构造的 `llama_sampler_init_dist(seed)`, 并在比较阶段新建相同 seed 的 sampler; 不能把命令行添加 `--temp 0` 当作它们已变成 greedy 测试。Test 4/5 在清空 memory 前已有 logits, 不能用其成功结果替代新 context 的正确恢复流程。

### 明确 checkpoint 位于哪一步

当前 `state_write_data()` 写入架构标识和 memory, 没有写入 logits/embedding 输出 buffer 或外部 sampler RNG。头文件的部分注释仍提到 logits/embedding, 本周以实际读写函数与测试为证据。

本周统一采用“最后一个已知输入尚未 decode”的 checkpoint:

```text
prompt = t[0] ... t[P-2] t[P-1], P >= 2

before saving:
  memory contains positions 0 ... P-2
  application records pending_token = t[P-1], pending_pos = P-1

baseline and restored branch both:
  decode(pending_token, pending_pos, output=true)
  -> fresh logits -> greedy sample -> decode sampled token -> continue
```

- `common_prompt_batch_decode()` 的保存路径也会先处理前 `P-1` 个 token, 再保存, 最后处理末 token。文件保存的 token 列表可以包含全部 `P` 个 token, 因此文件 token 数不必等于当时 memory 中的 token 数。
- 恢复后不要直接读旧 logits, 也不要无条件把已经存在的最后一个位置再 decode 一遍。若采用另一种 checkpoint 边界, 必须重新说明输入重放或回退的合法性。
- 保存原始字节数; 恢复时不能在空 context 上重新调用 `get_size()` 来决定旧状态的长度。
- 应用另存完整 token 历史、pending token/position、目标 sequence 映射、生成计数、模型文件标识及 context 参数。greedy 基线不涉及 RNG; 随机 sampler 还要处理其状态和接受历史, 仅重设 seed 不一定能续接中途采样。
- 普通 host state 用于本周持久化。扩展阅读 `PARTIAL_ONLY` 与 `ON_DEVICE` flags: 前者不是完整 memory 快照; 后者依赖 backend 中保存的数据, 同一 sequence 的后续 state 获取可使先前保存的 device 数据失效。

### 当天输出与验收

- API 表、五段测试的真实断言、checkpoint 时间线与应用侧字段清单。
- 能解释文件加载成功与模型/参数完全匹配之间的区别。实验主动固定同一模型、build、backend、KV 类型与 context 参数。
- 能指出自己的恢复实验中哪次 decode 产生了随后用于采样的 logits。

## Day 5: recurrent/hybrid memory 与能力边界

**投入: 2 小时。** 将同一套接口放到另一种状态表示上, 检查哪些普通 KV 结论不能照搬。

### 学习路线

1. **0-10 分钟**: 闭卷解释普通 attention 为什么需要历史 K/V。
2. **10-45 分钟**: 阅读 [`src/llama-memory-recurrent.h`](../../src/llama-memory-recurrent.h) 和 [`src/llama-memory-recurrent.cpp`](../../src/llama-memory-recurrent.cpp)。定位 `r_l/s_l`、cell 的 `src/tail/seq_id`、`seq_rm()` 与 `n_rs_seq` 相关分支。
3. **45-75 分钟**: 阅读 [`src/llama-memory-hybrid.cpp`](../../src/llama-memory-hybrid.cpp) 的 `init_batch()`、`seq_rm()`、`get_can_shift()` 和 state 读写。解释为什么 `seq_rm()` 先询问可能失败的 recurrent 分支, 再修改 attention 部分。
4. **75-100 分钟**: 完成下表, 每个结论至少关联一个函数或字段。

   | 对比项 | 普通 KV | recurrent | hybrid |
   | --- | --- | --- | --- |
   | 主要数据 | 历史位置的各层 K/V | recurrent 状态及管理 metadata | attention 与 recurrent 两部分 |
   | cell 的主要含义 | 历史 token 的物理槽位 | 状态槽位及序列关联, 不等于每个历史 token 各存一份 K/V | 两套表示需要协调 |
   | 部分删除/回退 | 普通路径可删除位置区间 | 受状态表示和已保留回退状态限制, 可能返回 false | 需满足两部分约束 |
   | position shift | 有 RoPE 时还需更新 K | 当前 `get_can_shift()` 返回 true, 移动 position 不等于重新计算历史状态 | 当前能力判断取决于 attention 部分 |
   | batch 准备 | 按 stream/slot 布局分配 | 受 recurrent sequence 布局约束 | 需要同时准备两部分状态 |
   | state | metadata 与各层 K/V | recurrent metadata 与实际状态 | 默认保存两部分, flags 可以改变范围 |

5. **100-120 分钟**: 为周末实验预写“操作 -> 预期 -> 证据”表, 准备共享分叉、移除隔离、单序列 shift、state 恢复四组用例。

### 当天输出与验收

- `notes/week5-day5.md`: 三类 memory 对照表, 一组 partial remove 失败条件, 周末实验预期。
- 不能用“recurrent 永远不能回退”概括所有配置; 当前实现存在受 `n_rs_seq` 等条件约束的回退状态机制。
- 能说明统一接口没有承诺相同的数据布局、同样的复制代价或任意范围的删除能力。

## Day 6: 共享前缀、分支隔离与 context shift 实验

**投入: 4 小时。** 时间分配: 回忆 10 分钟, 环境准备 25 分钟, GDB 观察 40 分钟, 独立分叉实验 105 分钟, shift 实验 45 分钟, 整理 15 分钟。

### 1. 准备环境和结果目录

沿用已配置的 CPU Debug build。若尚未配置, 先按总计划的构建章节完成; 以下只构建本周已有目标。

```bash
export LLAMA_REPO=/home/king/llama.cpp
export STUDY_DIR=/home/king/llama-study
export MODEL="$LLAMA_REPO/deepseek-r1-1.5b-f16.gguf"
export W5_RESULTS="$STUDY_DIR/results/week5"
cd "$LLAMA_REPO"
mkdir -p "$STUDY_DIR/notes" "$STUDY_DIR/experiments" \
  "$W5_RESULTS/day6" "$W5_RESULTS/day7/state" "$W5_RESULTS/day7/parallel"
test -f "$MODEL"
git rev-parse --short HEAD > "$W5_RESULTS/commit.txt"
cmake --build build-study-debug -j 4 \
  --target llama-parallel test-save-load-state test-state-restore-fragmented
./build-study-debug/bin/llama-parallel --help \
  > "$W5_RESULTS/parallel-help.txt" 2>&1
./build-study-debug/bin/test-save-load-state --help \
  > "$W5_RESULTS/state-help.txt" 2>&1
```

保存模型身份和有效 context/KV 参数。后续不能只凭文件名或架构字符串认定两次运行使用了相同模型。

### 2. 用已有 parallel 示例观察共享关系

阅读 [`examples/parallel/parallel.cpp`](../../examples/parallel/parallel.cpp) 的 system prompt prefill、`seq_cp()` 和请求结束时的 `seq_rm()`。显式使用 `-kvu -pps`: 前者启用 unified KV, 后者启用此示例的前缀共享。

```bash
gdb --args ./build-study-debug/bin/llama-parallel \
  -m "$MODEL" -ngl 0 -fa off -c 2048 -b 1024 -ub 128 \
  -t 4 -tb 4 -np 2 -ns 4 -n 4 -kvu -pps \
  --temp 0 --seed 1234 -p "Name one color."
```

进入 GDB 后先用 `start` 停在 main, 让共享库符号加载完成, 再观察首次同 stream copy:

```gdb
set pagination off
set print elements 16
start
break src/llama-kv-cache.cpp:llama_kv_cache::seq_cp
continue
bt 6
print seq_id_src
print seq_id_dst
print p0
print p1
print n_stream
print seq_to_stream[seq_id_src]
print seq_to_stream[seq_id_dst]
set $w5_cache = this
print $w5_cache->v_cells[0].pos[0]
print $w5_cache->v_cells[0].seq[0]
finish
print $w5_cache->v_cells[0].pos[0]
print $w5_cache->v_cells[0].seq[0]
```

本例保留 seq 0 保存 system prompt, 两个客户端实际使用 memory seq 1、2。首次 copy 预期位置不变、归属集合增加, 不新增整份前缀 K/V。`-ns 4` 表示总请求数; 日志里的请求序号不能直接当作 memory sequence ID。

补充断点时按问题逐个启用: `llama_kv_cache::find_slot`、`llama_kv_cache::apply_ubatch`、`llama_kv_cache::seq_rm`。在 `apply_ubatch()` 先用调用栈区分 Day 2 的临时规划和真实应用。

需要文本 cell 图时, 可以对短程实验设置 `LLAMA_KV_CACHE_DEBUG=3`, 并让 common 工具开启 `-v`。该路径可显示位置和归属概况, `.` 表示空 cell, `M` 表示多个 sequence 归属; 它会增加大量日志, 不用于性能测量。

### 3. 独立实验: 共享前缀后分叉, 与独立 prefill 对照

在第 1 周最小程序上增加实验模式, 文件建议为 `experiments/week5-memory-study.cpp`, 复用其构建方式。开始编码前先写出 batch 字段、每个 sequence 的 `next_pos` 和以下断言。

统一配置建议: `n_ctx=256,n_batch=64,n_ubatch=64,n_seq_max=2,kv_unified=true`, CPU、F16 K/V、关闭 Flash Attention。实际生效值仍以 context 日志为准。前缀控制在 8-16 个 token, 私有后缀不超过 4 个 token, 每个分支最多生成 4 个 token; token ID 由 tokenizer 产生并记录。

1. 在 seq 0 prefill 公共前缀 `P`, 用显式 position `0..P-1`。`llama_synchronize(ctx)` 后调用 `llama_memory_seq_cp(mem,0,1,0,P)`; seq 1 此前为空。
2. 检查两条序列的 position 范围一致。用 cell metadata 证明前缀增加了 seq 1 归属且 used cells 没有翻倍, 不能仅凭两次文本相似就认定共享成功。
3. 分别向 seq 0 输入后缀 A、向 seq 1 输入不同后缀 B, 起始 position 都为 `P`。使用独立的 `next_pos` 和 `llama_sampler_init_greedy()` 实例。
4. 先让一条分支完成短续写, 再运行另一条。每次采样前确保刚刚 decode 的最后输出属于当前分支; `seq_cp()` 本身不产生新 logits。为简化索引, 每次 API batch 只含一个 sequence, 只请求最后一个 token 的输出。
5. 使用新 context 做 A、B 各自的独立 prefill 基线, 保持相同模型、参数、前缀/后缀分块与生成停止规则。比较 token ID 数组, 不只比较可见文本。
6. 重建初始条件, 交换两条分支的运行顺序。两种顺序都应与各自独立基线一致; 后缀不同不保证生成 token 必然不同, 所以“分支输出不相等”不能作为隔离断言。
7. 增加删除对照: 在相同边界保存 seq 1 的 host state 字节, 删除 seq 0, 再保存 seq 1。比较同一个 seq ID 的前后字节, 并检查共享前缀仍由 seq 1 持有。随后与未删除 seq 0 的控制运行提交相同的待输入 token, 产生新 logits 后比较 seq 1 的续写。

所有 memory 修改都放在前一次计算已经同步完成的稳定边界。检查每次 `llama_decode()` 返回值, `seq_rm()` 的 bool 和 state 的字节数。实验先采用顺序执行; 两条分支交替 batching 属于选做, 必须额外处理每条分支的输出索引。

| 必做对照 | 预期 | 证据 |
| --- | --- | --- |
| 共享前后 | 前缀物理 cells 不翻倍, 归属增加 | cell 图、used cells |
| shared A 与独立 A | 相同 token IDs | 逐步 token/position 表 |
| shared B 与独立 B | 相同 token IDs | 同上 |
| A 先运行与 B 先运行 | 各分支输出不受运行顺序影响 | 两轮对照日志 |
| 删除 A 前后 | B 的 host state 字节不变, 后续生成一致 | 字节比较与 fresh decode 后的 token 对照 |

如果 token 不一致, 记录第一个分歧位置和对应 logits, 依次检查模型/参数、prompt tokenization、输入分块、sequence 归属、position、输出索引与 sampler。不要先把所有差异归因于浮点误差。

### 4. 独立实验: 单 sequence 的 context shift

在新建或清空的 context 中运行 Day 3 的 8-token 例子, 不复用共享前缀的 cells:

1. prefill 8 个合法 token, 显式 position `0..7`, 同步后记录 cell 与 position。
2. 检查 `llama_memory_can_shift()`。若不支持, 记录具体实现分支, 换回本周普通 Transformer 基线。
3. 执行 `seq_rm(0,2,5)` 与 `seq_add(0,5,8,-3)`; 检查删除返回值, 将应用的 `next_pos` 改为 5。
4. 在 `llama_kv_cache::update` 与 `llama_kv_cache::build_graph_shift` 观察 pending shift 的消费。提交一个已知 token, position 为 5, 请求输出并同步。
5. 验证位置变化、可用槽位和下一次 decode 成功; 记录更新前后 pending shift 的变化, 并解释为何 `seq_add()` 不等于简单移动 K/V 字节。

这里通过主动删除和移动触发相同机制, 无需用长生成等待 cache full。生成质量变化和不同 context 策略的评估留作扩展; 本实验不要求与删除文本后重新 prefill 的数值相同。

### 当天输出与验收

- 实验参数、共享前后 cell 图、五组对照结果、shift 断点记录。
- `experiments/week5-memory-study.cpp`: 能独立解释输入字段、每次 memory 操作和每个断言的最小实验。
- 能证明共享减少了重复前缀状态、私有后缀保持隔离, 并解释删除一个 owner 后共享 cell 的存活条件。

## Day 7: 保存、清空、恢复与第 5 周验收

**投入: 4 小时。** 时间分配: 回忆 10 分钟, 独立恢复实验 90 分钟, 现有测试 45 分钟, parallel 对照 35 分钟, 整理图表 35 分钟, 闭卷验收 25 分钟。

### 1. 独立恢复实验: checkpoint 后的 token 必须一致

复用 Day 6 程序, 增加 host sequence state 模式。沿用 Day 4 的“最后一个已知输入尚未 decode”约定, 先做 same-context clear/restore, 再做新 context 恢复。

1. 准备长度至少为 2 的确定 token 列表, seq 0 只 decode 前 `P-1` 个 token。同步后保存 seq 0 的 host state: 先取 size, 分配 byte vector, 再 `get_data()` 并检查实际字节数。
2. 另存 pending token `t[P-1]`、pending position `P-1`、完整 token 列表和模型/context 参数。此时预期 seq 0 的 `pos_max=P-2`。
3. 基线运行 decode pending token, 请求输出, 再用全新 greedy sampler 生成最多 4 个 token。统一约定每个采样结果都再 decode 一次, 记录 token IDs、position 和停止原因。
4. 同步, 调用 `llama_memory_clear(mem,true)`。检查目标 sequence 的 `pos_min/max` 为 `-1`, 不把旧 logits 当成已清空 memory 的输出。
5. 使用原始 byte vector 和原始长度恢复到 seq 0, 检查返回字节数与 position 范围。恢复应用变量, 再 decode 同一个 pending token, 用新 greedy sampler 按同一停止规则生成, 与基线逐 token 比较。
6. 新建相同配置的 context, 保持 `n_seq_max=2,kv_unified=true`, 将同一份状态恢复到合法且为空的 seq 1。后续所有 batch 都明确使用 seq 1; 再 decode pending token、继续生成并与基线比较。
7. 对 full-context 文件模式复核 token 列表、magic/version 与加载返回值, 用下一节已有测试覆盖该路径。不要把 sequence raw bytes 传给完整 context 的文件加载 API。

调试时在 `llama_context::state_seq_write_data/state_seq_read_data` 和 `llama_kv_cache::state_write/state_read` 分别设断点, 观察 seq ID、flags、选中的 cell 数以及恢复目标。沿用 Day 6 的 `start` 和源码限定断点方式, 把接口包装与实际 tensor 读写区分开。

| 运行 | checkpoint 中的最后位置 | 恢复目标 | 采样前的新 decode | 验收 |
| --- | ---: | --- | --- | --- |
| baseline | `P-2` | 无 | `t[P-1]` at `P-1` | 保存参考 token IDs |
| clear/restore | `P-2` | 同 context, seq 0 | 相同 pending token/position | IDs、停止原因一致 |
| new context | `P-2` | 新 context, seq 1 | 相同 token/position, seq 改为 1 | IDs、停止原因一致 |

state 字节比较适合证明同一 seq 在特定操作前后未改变; 不要求不同目标 ID 或不同物理布局的序列化结果永远字节相同。恢复语义的主要证据是正确的 metadata、已消费字节数与受控续写。

### 2. 运行现有 state 测试

这些命令沿用 Day 6 的环境变量。`test-save-load-state` 当前将状态写到工作目录下的 `dump_state.bin`, 因此在独立结果目录运行; 不添加该测试未提供的 `-o` 参数。

```bash
(
  cd "$W5_RESULTS/day7/state" || exit 1
  "$LLAMA_REPO/build-study-debug/bin/test-save-load-state" \
    -m "$MODEL" -ngl 0 -fa off -c 256 -b 16 -ub 16 \
    -t 4 -tb 4 -n 2 --seed 1234 \
    > save-load.stdout.txt 2> save-load.stderr.txt
)
```

未传 prompt 时, 该测试按 seed 生成 `n_batch` 个随机合法 token, 因此这里使用 16-token 输入。验收为退出码 0、stdout 中全部测试通过; 同时核对 Day 4 的断言表, 不把内置 dist sampler 结果写成自建 greedy 实验结果。

```bash
"$LLAMA_REPO/build-study-debug/bin/test-state-restore-fragmented" \
  -m "$MODEL" -ngl 0 -fa off -b 256 -ub 128 -t 4 -tb 4 \
  > "$W5_RESULTS/day7/state/fragmented.stdout.txt" \
  2> "$W5_RESULTS/day7/state/fragmented.stderr.txt"
```

阅读 [`tests/test-state-restore-fragmented.cpp`](../../tests/test-state-restore-fragmented.cpp) 后画出 `0,1,2` 三条序列交错占位的布局。它默认使用 unified KV、3 条序列和 256 context, 每条输入 70 个 token; 不随意覆盖这些参数, 以免破坏测试针对的碎片分配条件。

验收为退出码 0、保存/恢复字节数一致、恢复后 decode 成功。这个测试验证非连续空闲 cells 的恢复和可继续计算, 没有与独立基线逐 token 或逐 logits 比较, 不能替代上一节的一致性实验。

本周直接给两个测试传入本地完整模型。对应 CTest 配置可能包含下载模型的 fixture, 不把直接运行二进制与运行所有 CTest 当成同一个动作。

### 3. 完成 parallel 小实验并解释日志

```bash
"$LLAMA_REPO/build-study-debug/bin/llama-parallel" \
  -m "$MODEL" -ngl 0 -fa off -c 2048 -b 1024 -ub 128 \
  -t 4 -tb 4 -np 2 -ns 4 -n 4 -kvu -pps \
  --temp 0 --seed 1234 -p "Name one color." \
  > "$W5_RESULTS/day7/parallel/parallel.stdout.txt" \
  2> "$W5_RESULTS/day7/parallel/parallel.stderr.txt"
```

- `-np 2` 为两个客户端; 示例额外保留 seq 0, 因此 context 为 system sequence 和两个 client sequences 准备容量。
- `-ns 4` 为四个请求, 不是四个并发 memory sequences。追踪 client ID、请求序号与实际 batch seq ID 三者的映射。
- `-p` 控制请求 prompt, 不替换源码内置的 system prompt。启用共享时, system prompt 初始 prefill 在一次 decode 调用中提交, 所以必须给足 `n_batch`; 本地模型实测 system prompt 为 274 tokens, 此处取 `-b 1024`。
- 记录四个请求完成情况、共享前缀的 copy、结束请求的清理和下一请求的复用。不用这个 Debug 小实验作吞吐性能结论, 也不以四条输出“看起来合理”替代分支隔离验证。

编写本文时, 上述两个 state 测试与该 parallel 命令已在本地 Qwen2 F16 模型、CPU Debug build 上运行通过。自建共享/shift/恢复实验仍是本周需要亲手完成的学习任务。

### 4. 第 5 周验收清单

- [ ] 能解释 `llama_memory_i` 和 `llama_memory_context_i` 的职责, 区分规划、应用、graph 写入。
- [ ] 能画 token/position/cell/sequence/stream 的映射, 解释 used cells 与 `n_kv` 的区别。
- [ ] 能说明同 stream 的前缀共享与跨 stream 的实际复制有什么不同。
- [ ] 能手算 `seq_cp/rm/keep/add/div` 的状态变化, 并指出范围和能力限制。
- [ ] 能解释共享 cell 删除一个 owner 后是否仍占用, 不把 cell 回收等同于 buffer 释放。
- [ ] 能解释 context shift 的删除区间、位置变化、待执行 K 更新与应用 `next_pos` 更新。
- [ ] 已用独立基线和不同运行顺序验证两个共享分支的 greedy 续写与删除隔离。
- [ ] 能写出 checkpoint 的准确位置, 指出恢复后第一次产生新 logits 的 decode。
- [ ] 已验证 save/clear/restore 与新 context 的合法目标 seq 恢复, token IDs 和停止规则一致。
- [ ] 两个现有 state 测试通过, 并能说明各自实际断言及未覆盖内容。
- [ ] 已完成 `-np 2 -ns 4` parallel 实验, 能区分 client、request 与 memory sequence ID。
- [ ] 能说明 recurrent/hybrid 的状态表示、部分删除与 shift 能力边界, 并解释为何需要统一 memory 接口。

## 本周完成包

```text
/home/king/llama-study/
  experiments/
    week5-memory-study.cpp
  notes/
    week5-day1.md ... week5-day7.md
    week5-memory-interface.md
    week5-kv-cell-sequence-map.md
    week5-sequence-operations.md
    week5-context-shift.md
    week5-checkpoint-contract.md
    week5-recurrent-hybrid.md
    week5-review.md
  results/week5/
    commit.txt
    parallel-help.txt
    state-help.txt
    day6/
      shared-prefix.log
      branch-token-comparison.csv
      removal-isolation.log
      shift.log
      gdb-memory.txt
    day7/
      restore-token-comparison.csv
      checkpoint-metadata.txt
      state/
        dump_state.bin
        save-load.stdout.txt
        save-load.stderr.txt
        fragmented.stdout.txt
        fragmented.stderr.txt
      parallel/
        parallel.stdout.txt
        parallel.stderr.txt
```

主题笔记从每日记录中整理, 不重复抄写源码。进入第 6 周前, 应能把“恢复 memory 后如何获得正确 logits”讲清楚, 再研究 sampler 状态、采样链与随机生成的复现。
