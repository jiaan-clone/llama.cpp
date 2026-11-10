# week3 day6

本文对应 [第 3 周学习路线 Day6](../../003-llama-cpp-week3-daily-learning-route.md) 的输出与验收: 命令、模型身份和参数、预测/实测对照、差异解释及原始证据归档。

**按本次要求, 所有手动实验暂时跳过。本文已整理源码结论、待执行命令、预测表和证据要求; 没有运行架构测试、GDB、真实 GGUF 或 eval callback, 因而 Day6 的运行证据验收尚未完成。** 下文“待补”不是测试通过, 候选差异也不是已经观察到的现象。

当前源码基准为 `dd545f7e7`, 由本次只读仓库检查取得。没有为此创建 `results/week3/day6/` 的新日志或占位结果文件。将来执行实验时应重新记录当时的 commit。

## 习题1: 模型身份、实验目的与参数

### 1. 区分证据来源

| 来源 | 已知内容及依据 | 能证明什么 | 本次状态 |
| --- | --- | --- | --- |
| Llama 源码与纸面预测 | `src/models/llama.cpp` 和公共 helper | 公式、构图分支、预期 shape/op/依赖 | 已完成静态阅读, 详见 Day4/Day5 |
| synthetic Llama | `test-llama-archs --arch llama --seed 1234` 的源码生成 metadata, 用初始化函数提供权重 | 执行后可证明进入真正的 Llama 构图路径, 检查结构与相关测试结果 | 未运行 |
| 本地 Qwen2 GGUF | 第 2 周记录的 `deepseek-r1-1.5b-f16.gguf` | 执行后可观察真实模型的公共 helper/op/shape, 不能冒充真实 Llama 结果 | 本次未读取/运行, 沿用既有身份记录 |
| 真实 Llama GGUF, 若后续使用 | 需补文件路径、`general.architecture` value 和实际配置 | 执行后可提供真实 Llama 的运行证据 | 未指定本次实验文件, 待补 |

本地文件名虽然包含 DeepSeek-R1, 学习路线与 [第 2 周记录](../week2/01.%20week2-day1/01.%20week2-day1.md) 按 Qwen2 处理。旧记录同时注明, 当时的 `llama-gguf` 日志只打印 metadata key, 没有打印 `general.architecture` 的具体 value; 本次没有补做读取, 后续应保留能确认架构 value 的证据。

### 2. 可提前填写的配置, 与必须实测的值

| 项目 | synthetic Llama 的源码配置 | 本地 Qwen2 的历史记录 |
| --- | --- | --- |
| D / H / KvH / Dh | `256 / 2 / 2 / 128` | `1536 / 12 / 2 / 128` |
| I / V / L | `384 / 128 / 2` | `8960 / 151936 / 28` |
| token 与 ubatch 设置 | 测试生成 128 个 token; decode context 设置 `n_ubatch=64` | 待执行命令用 `-p "Hello llama" -b 32 -ub 32`; 实际 token 数待日志确认 |
| 输出请求 | 测试的 `common_batch_add(...,true)` 请求各 token 输出 | 由实际 batch/output 设置决定, O 待确认 |
| Flash Attention | 测试使用 context 设置, 运行时还需核对实际 `cparams.flash_attn` | 下文命令显式 `-fa off` |
| 证据性质 | 源码配置, 不是本次断点观测 | 历史记录代入, 不是本次 GGUF 实测 |

不要把测试的总输入 128 当成每个 graph 的 N: 实际 ubatch 会受 `n_ubatch` 和处理路径影响, 以 `process_ubatch()` 的 `ubatch.n_tokens` 为准。S、Kcap、O 也应按每次图记录, Kcap 不直接等于 S 或 `n_ctx`。

合成测试还需要留意: 当前虚拟加载路径可以按有效 shape 创建可选 tensor, 例如 bias; `TENSOR_SKIP_IF_VIRTUAL` 则让合并 QKV 的尝试返回空。不能将 Day5“无 bias”的纸面简化直接当成合成测试的固定节点清单。该测试的默认 `H=KvH=2` 是 MHA, 也不能据此声称已经实测了 GQA 的 head 分组共享。

合成权重适合构图、后端和保存/加载路径检查, 不证明真实语言模型质量, 也不替代与参考实现的逐层数值对齐。测试遍历 Dense 和 MoE 以及不同设备配置, 本周先跟踪 Dense; 实际输出中的 `SKIP` 必须保留为跳过。

## 习题2: 待执行命令与观察方法

**本节全部命令仅供后续实验使用, 本次未执行。** 当前未检查已有构建是否满足 Debug、CPU 和相关 target 的要求, 也没有确认下面命令能在本机直接运行成功。

### 1. 准备与架构测试

在仓库根目录操作。若尚未配置对应构建目录, 可按以下参数准备 CPU Debug 构建; 已有构建则先确认配置。

```bash
cd /home/king/llama.cpp
export STUDY_DIR=/home/king/llama-study
mkdir -p "$STUDY_DIR/results/week3/day6"
git rev-parse --short HEAD > "$STUDY_DIR/results/week3/day6/commit.txt"

cmake -S . -B build-study-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build-study-debug --target llama-eval-callback test-llama-archs -j

./build-study-debug/bin/llama-eval-callback --help \
  > "$STUDY_DIR/results/week3/day6/eval-help.txt" 2>&1
./build-study-debug/bin/test-llama-archs --arch llama --seed 1234 \
  > "$STUDY_DIR/results/week3/day6/test-llama-archs.log" 2>&1
printf '%s\n' "$?" > "$STUDY_DIR/results/week3/day6/test-llama-archs.exit-code.txt"
```

运行后记录设备、Dense/MoE、各项 OK/FAIL/SKIP 和退出码, 不仅摘取成功行。构建和运行失败时保留实际错误, 不能按预期填写结果。

### 2. GDB: 先进入实际 ubatch 处理

```bash
gdb --args ./build-study-debug/bin/test-llama-archs --arch llama --seed 1234
```

第一阶段只在实际 ubatch 入口停住, 避免先被初始化中的图预留/探测误导:

```gdb
set pagination off
set breakpoint pending on
set logging file /home/king/llama-study/results/week3/day6/gdb-llama.txt
set logging overwrite on
set logging enabled on
break src/llama-context.cpp:llama_context::process_ubatch
run
bt 8
print ubatch.n_tokens
print ubatch.n_seqs
print this->model.arch
print this->cparams.flash_attn
```

确认是 synthetic Llama 的 Dense 路径后, 再设置构图和执行入口断点。以下 callback 条件同时覆盖 layer 0 和 `il=-1` 的全局节点, 用于观察 embedding、最终 norm 和 logits:

```gdb
break src/models/llama.cpp:llama_model_llama::build_arch_graph
break src/llama-graph.cpp:llm_graph_context::cb if il == 0 || il == -1
break src/llama-context.cpp:llama_context::graph_compute
continue
```

使用带源文件的符号断点, 减少误停在动态库 `@plt` 入口而缺少局部变量的情况。动态库尚未加载时可以先保留 pending breakpoint。后续在每次停住时先确认所在函数, 再执行对应的观察命令。

| 停点 | 要记录的内容 |
| --- | --- |
| `build_arch_graph()` | `bt 6`, 确认调用链来自本次 ubatch; 进入 graph constructor 对照 Day4 顺序 |
| 构图 `cb()` | `name`、`il`、tensor 地址、`op/ne/nb/src[]`、`view_src/view_offs` |
| `graph_compute()` | graph 的 node/leaf 数、实际 Flash Attention 设置、完整节点列表 |

在 `cb()` 入口执行:

```gdb
print name
print il
print cur
print cur->name
print cur->type
print cur->op
print cur->ne
print cur->nb
print cur->src
print cur->view_src
print cur->view_offs
```

`name` 是准备赋予的语义名, `cur->name` 此时可能尚未更新。这个停点用于检查构图描述, 不读取尚未执行的激活 `data` 来证明数值正确。Q 的 `MUL_MAT`、可选 `ADD`、`RESHAPE/VIEW`、`ROPE` 要分别记录, 不能用重复的 `Qcur-0` 名称混为一条。

在 `graph_compute()` 入口执行:

```gdb
print gf->n_nodes
print gf->n_leafs
print this->cparams.flash_attn
print batched
set $i = 0
while $i < gf->n_nodes
  set $t = gf->nodes[$i]
  printf "%d %p %s op=%d ne=[%lld,%lld,%lld,%lld]\n", $i, $t, $t->name, $t->op, $t->ne[0], $t->ne[1], $t->ne[2], $t->ne[3]
  set $i = $i + 1
end
```

选定关键节点后再记录其 `nb/src/view_src/view_offs`。`ggml_cgraph` 内部定义在 `ggml/src/ggml-impl.h`; 若调试信息没有暴露结构, 可尝试公共 `ggml_graph_n_nodes(gf)` 和 `ggml_graph_node(gf,i)` 接口。

如果本次复用了已有图, 不一定再次经过 builder 或构图 `cb()`; 仍可在 `graph_compute()` 看最终图。layer 0 与全局节点之外的最后一层选行, 可从完整 graph 列表查找相应 `GET_ROWS` 及依赖, 需要时再补该层断点。

### 3. 真实 GGUF 的非 FA callback

下面沿用历史 Qwen2 文件路径, 如果之后换成确认架构为 Llama 的完整 GGUF, 需同步更新模型身份和配置记录。

```bash
cd /home/king/llama.cpp
export STUDY_DIR=/home/king/llama-study
export MODEL="$PWD/deepseek-r1-1.5b-f16.gguf"
test -f "$MODEL"

./build-study-debug/bin/llama-eval-callback \
  -m "$MODEL" -ngl 0 --device none --no-op-offload \
  -fa off -c 128 -b 32 -ub 32 -t 4 -tb 4 \
  -p "Hello llama" \
  > "$STUDY_DIR/results/week3/day6/eval.stdout.txt" \
  2> "$STUDY_DIR/results/week3/day6/eval.stderr.txt"
printf '%s\n' "$?" > "$STUDY_DIR/results/week3/day6/eval.exit-code.txt"

rg -n 'common_debug_cb_eval:' \
  "$STUDY_DIR/results/week3/day6/eval.stdout.txt" \
  "$STUDY_DIR/results/week3/day6/eval.stderr.txt" \
  > "$STUDY_DIR/results/week3/day6/nodes.txt"

rg 'embd|Qcur-0|Kcur-0|Vcur-0|attn_norm-0|kq-0|kq_soft_max-0|kqv-0|ffn_.*-0|l_out-0|result_norm|result_output' \
  "$STUDY_DIR/results/week3/day6/nodes.txt" \
  > "$STUDY_DIR/results/week3/day6/layer0-and-output.txt"
```

以上 `-ngl 0 --device none --no-op-offload` 用于 CPU 路径, `-fa off` 用于展开非 FA attention, `-b/-ub` 控制 batch/ubatch 大小。实际 N 以日志中的 `number of input tokens` 及 ubatch 记录为准; 应确认短 prompt 不超过 32 个输入 token, 不凭字符数猜 token 数。

`examples/eval-callback/eval-callback.cpp` 的 `run()` 只调用一次 `llama_decode()`, 没有逐 token 生成循环, 并关闭 warmup。一次 decode 仍可能被拆成多个 ubatch; 不能用 `-n` 将这个工具解释成多轮生成实验。

### 4. 执行 callback 能提供哪些证据?

`common_debug_cb_eval()` 在 `ask=true` 时接受观察请求, 在 `ask=false` 阶段收集执行结果。日志会给出名称、type、op 描述、部分源 tensor 信息和 `ne[]`; 数值打印按 `nb[]` 访问。

完整 `src[]`、stride 和 view 信息仍应通过 GDB 补全。构图 `cb()` 是命名/放置 hook, 执行 callback 才用于观察已执行的数据, 二者不能混用。读取、同步和打印会影响开销, 本实验的耗时不能作为 benchmark。

## 习题3: 预测 / 实测对照表

下面先填普通 dense、无额外 bias/scale 的公式主干; 若实测路径存在 bias、合并 QKV 或 FA, 应按实际分支展开。N/S/Kcap/D/H/KvH/Dh/I/O/V 沿用 Day5, `Nt=N` 表示普通层, 最后一层选行后 `Nt=O`。

**所有“实测及证据”栏当前均未填写。** 后续每行至少附日志文件及行号, 或 GDB 停点、图节点索引与指针; 同时记录对应的模型身份和本次 ubatch, 避免将 synthetic Llama 与 Qwen2 的记录混成一条证据。

| 编号 / 关键阶段 | 公式与源码入口 | 预期 op | 预测 GGML `ne[]` | 实测及证据 |
| --- | --- | --- | --- | --- |
| 1. embedding | `E[ids]`, `build_inp_embd()` | `GET_ROWS` | `[D,N]` | 未运行, 待补 |
| 2. attention norm | RMS 归一化后乘 gamma, `build_norm()` | `RMS_NORM + MUL` | `[D,N]` | 未运行, 待补 |
| 3. Q projection | `Xn Wq^T`, `build_qkv()` | `MUL_MAT`, 可选 bias `ADD` | `[H*Dh,N]` | 未运行, 待补 |
| 4. K projection | `Xn Wk^T`, `build_qkv()` | `MUL_MAT`, 可选 bias `ADD` | `[KvH*Dh,N]` | 未运行, 待补 |
| 5. V projection | `Xn Wv^T`, `build_qkv()` | `MUL_MAT`, 可选 bias `ADD` | `[KvH*Dh,N]` | 未运行, 待补 |
| 6. 拆 head | `build_qkv()` 的 reshape 或合并投影 view | `RESHAPE/VIEW` | Q `[Dh,H,N,1]`, K/V `[Dh,KvH,N,1]` | 未运行, 待补 |
| 7. Q/K RoPE | 按 position 旋转, Llama graph | `ROPE` | 与拆 head 后相同 | 未运行, 待补 |
| 8. score | `Q K^T`, `build_attn_mha()` | 非 FA 为 `MUL_MAT` | `[Kcap,N,H,1]` | 未运行, 待补 |
| 9. P | `softmax(score/sqrt(Dh)+mask)` | 非 FA 为 `SOFT_MAX` | `[Kcap,N,H,1]` | 未运行, 待补 |
| 10. 加权 V | `P V`, `build_attn_mha()` | 非 FA 为 `MUL_MAT` | `[Dh,N,H,1]` | 未运行, 待补 |
| 11. Wo | 拼接 heads 后乘 `Wo^T`, `build_attn()` | `PERMUTE/CONT + MUL_MAT` | `[D,N]` | 未运行, 待补 |
| 12. 第一次 residual | `R=X+Attn`, Llama graph | `ADD` | `[D,Nt]` | 未运行, 待补 |
| 13. FFN norm | `RMSNorm(R)`, `build_norm()` | `RMS_NORM + MUL` | `[D,Nt]` | 未运行, 待补 |
| 14. gate/up | `Rn Wgate^T`、`Rn Wup^T`, `build_ffn()` | 两个 `MUL_MAT`, 可选后处理 | 各 `[I,Nt]` | 未运行, 待补 |
| 15. SwiGLU | `SiLU(gate).*up`, `build_ffn()` | `GLU/SWIGLU` | `[I,Nt]` | 未运行, 待补 |
| 16. down | 门控结果乘 `Wdown^T`, `build_ffn()` | `MUL_MAT`, 可选 bias/scale | `[D,Nt]` | 未运行, 待补 |
| 17. 第二次 residual | `Y=R+F`, Llama graph | `ADD` | `[D,Nt]` | 未运行, 待补 |
| 18. 最后一层选行 | Wo 之后对 Attn 和 X 使用相同 out ids | 两个 `GET_ROWS` | `[D,N] -> [D,O]` | 未运行, 待补 |
| 19. 最终 norm | `RMSNorm_out(Y_selected)` | `RMS_NORM + MUL` | `[D,O]` | 未运行, 待补 |
| 20. logits | `Ynorm Wout^T`, `build_lora_mm()` | `MUL_MAT`, 特殊输出参数另记 | `[V,O]` | 未运行, 待补 |

编号按观察项目排列, 不是 graph index; 第 18 项实际上发生在最后一层的第 12 项之前。节点名可能重复或被后续 `cb()` 改名, 不用 callback 行数或唯一名称数量直接推断计算节点总数。

还需至少补两组依赖/布局证据:

- 一个 matmul 的 `src[]`: 例如 Q 投影中 `src[0]=wq`, `src[1]=Xn`, 或 score 中 `src[0]=k`, `src[1]=q`。
- 一个换轴前后的 `ne[]/nb[]`: 与 [Day5 stride 推导](week3-day5.md) 对照, 同时记录 type、`view_src`、`view_offs`; 不将 F32 预测套到量化存储。

## 习题4: 至少三个差异及原因应如何记录?

**本次没有运行, 不能填写“已观察到三个差异”。** 以下列出后续应检查的候选现象、源码解释和所需证据。只有实际日志/断点支持之后, 才将对应条目标为已验证。

| 候选现象 | 可能原因及源码解释 | 需要补的证据 | 当前状态 |
| --- | --- | --- | --- |
| 公式有缩放、mask、softmax, 图中只有一个 softmax 节点 | `ggml_soft_max_ext()` 用一个 `SOFT_MAX` 表达三步; scale 在 op 参数中, mask 在 `src[1]` | 节点 op、mask 源、scale 参数或调用位置 | 待验证 |
| 公式中的 SiLU 和逐元素乘法未分别出现 | 普通 `build_ffn()` 调用 `ggml_swiglu_split()` | `GLU/SWIGLU` op 描述、gate/up 两个输入和 shape | 待验证 |
| `kq.ne[0]` 大于 prompt token 数 | N 是当前输入, S 还含有效历史; Kcap 又可能含 padding/空槽 | 同次 ubatch 的 N、有效位置、`n_kv`、mask 与 kq shape | 待验证 |
| synthetic 图没有独立的 kq/softmax/kqv | 实际启用了 Flash Attention, 用 `FLASH_ATTN_EXT` 表达 attention | 实际 `cparams.flash_attn`、最终 graph op 与依赖 | 待验证 |
| Qcur 出现 ADD 或重复同名节点 | 可选 bias, 或投影/reshape/RoPE 复用同一语义名; synthetic 虚拟加载也可能创建 bias | `wq_b` 是否非空、节点 op、shape、指针及来源 | 待验证 |
| 最后一层 FFN 的 token 维度变小, logits 为 `[V,O]` | 两条支路先按 out ids 选行, 后续仅计算 O 个位置 | N/O、out ids、两个 `GET_ROWS` 与最终 logits | 待验证 |
| cache helper 叫 cpy, 实际图为 SET_ROWS | 当前普通 cache 使用 `ggml_set_rows()` 写入 | `SET_ROWS` 的 op 和三个 `src[]` | 待验证 |

一个完整差异记录应写成“预测是什么、实际是什么、在哪条原始证据看到、源码为何允许这个结果、需要回填哪张预测表”。如果某条预期现象没有出现, 记录实际分支, 不为了凑数改写观测。

## 习题5: 证据归档与验收状态

计划归档到 `/home/king/llama-study/results/week3/day6/`, 下表只是目录约定, **不代表这些文件已由本次任务产生**。

| 计划文件 | 应保存的内容 | 本次状态 |
| --- | --- | --- |
| `commit.txt` | 实验执行时的 commit | 未新建; 当前源码 commit 已写在本文开头 |
| `eval-help.txt` | 实际二进制的参数帮助 | 未生成 |
| `test-llama-archs.log`、`test-llama-archs.exit-code.txt` | 架构测试完整结果与退出码 | 未生成 |
| `gdb-llama.txt` | 调用栈、模型身份、ubatch、关键 tensor 及图列表 | 未生成 |
| `eval.stdout.txt`、`eval.stderr.txt`、`eval.exit-code.txt` | 真实模型 callback 的原始输出与退出码 | 未生成 |
| `nodes.txt` | 带原始文件位置的 callback 节点摘录 | 未生成 |
| `layer0-and-output.txt` | layer 0 与最终输出的筛选摘录 | 未生成 |

实验记录还需写明模型文件及架构、配置来源、命令参数、设备、实际 N/S/Kcap/O、FA 开关和所观察的层。可以补记在本文对应表格中, 保留原始日志作为证据。

| Day6 验收要求 | 本次完成情况 |
| --- | --- |
| 命令、模型身份和参数记录 | 已整理待执行命令, 区分源码配置/历史记录/待实测字段 |
| 至少 10 个关键阶段的公式、shape、op 预测 | 已列出 20 个阶段 |
| 预测与实测对照 | 预测已填; 实测暂时跳过, 待补 |
| 至少 3 个实测差异及原因 | 已给出候选现象和原因; 实际差异与证据待补 |
| 架构测试、GDB、callback 原始日志归档 | 手动实验暂时跳过, 本次未生成 |
| “公式 + shape + op + 运行证据”闭环 | 尚未完成, 不能将源码阅读代替运行证据 |

学习路线在真实 GGUF 不可用时允许先保留 synthetic Llama 的 GDB 证据, 但本次要求暂时跳过手动实验, 所以 synthetic 与真实模型两条运行路径都保留为待完成。

## 源码核对入口

- [test-llama-archs.cpp](../../../../tests/test-llama-archs.cpp): 合成 metadata、初始化权重、Dense/MoE 遍历、token/batch 设置、测试结果状态。
- [eval-callback.cpp](../../../../examples/eval-callback/eval-callback.cpp): token 数打印、关闭 warmup、一次 `llama_decode()` 和 callback 注册。
- [common/debug.cpp](../../../../common/debug.cpp): `ask` 两阶段、op/shape 输出和数据读取。
- [llama-context.cpp](../../../../src/llama-context.cpp): `process_ubatch()`、构图复用、eval callback 注册和 `graph_compute()`。
- [Llama graph](../../../../src/models/llama.cpp)、[公共 graph helper](../../../../src/llama-graph.cpp): 预测表的公式、shape、op 与构图命名。
- [llama-model-loader.cpp](../../../../src/llama-model-loader.cpp): 无文件加载的可选 tensor 创建和 `TENSOR_SKIP_IF_VIRTUAL`。
- [ggml-impl.h](../../../../ggml/src/ggml-impl.h): GDB 图遍历所用 `ggml_cgraph` 内部结构。
