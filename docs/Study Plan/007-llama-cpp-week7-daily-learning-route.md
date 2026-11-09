# llama.cpp 第 7 周每日学习路线: GGML、CPU/GPU Backend、量化、内存与性能分析

本路线根据 [llama.cpp 8 周深度学习与二次开发计划](llama-cpp-8-week-learning-plan.md) 的第 7 周安排展开, 承接 [第 6 周: 词表、Tokenizer、Sampler、Grammar 和结构化输出](006-llama-cpp-week6-daily-learning-route.md)。本周回到 logits 产生之前, 把第 3 周的模型图、第 4 周的 decode 调度和第 5 周的 KV Cache 连接到实际硬件执行。

本周主线:

```text
model graph -> ggml_tensor / op / ggml_cgraph
  -> device capability + tensor buffer placement
  -> scheduler assignment / split / allocation
  -> input copy + backend graph compute + synchronization
  -> CPU MUL_MAT / optional CUDA dispatch
  -> F16 and block quantization
  -> PP / TG + memory measurements + quality observations
  -> evidence, bottleneck hypothesis, next experiment
```

完成标准: 能区分 tensor metadata、数据 buffer、device、backend 和 scheduler; 能追踪一个 `MUL_MAT` 从构图到 CPU 执行; 完成至少一组可重复的 CPU 性能对照, 用原始记录解释性能与内存差异, 明确量化质量结论的适用范围。

## 使用约定与本周边界

- 工作日每天 2 小时, 周末每天 4 小时, 共 18 小时。每天开头 10 分钟闭卷回忆, 结尾至少 15 分钟整理证据。
- 本文按工作区 `545ddb30a` 的源码核对。已有二进制可能来自不同构建, 学习前增量构建所需 target, 保存实际 `--help`、commit 和 CMake 配置; 函数用符号定位。
- CPU 是必做路径, CUDA 是选做对照。总计划记载过 CUDA driver/runtime 不匹配, 这只是当时的记录; 第 2 天重新确认。环境排查最多占用当天 20 分钟, 未解决就保留失败日志并继续 CPU 路线。
- 沿用本地 `deepseek-r1-1.5b-f16.gguf`, 实际架构为 Qwen2。也可使用完整的本地 Llama GGUF, 但同一组对照必须来自同一份浮点权重, 固定 tokenizer、模型配置和输入。词表专用 GGUF 不能用于性能实验。
- Debug 用于断点, Release 用于性能。保持 CPU 编译选项一致; 本周基线关闭 CUDA 和 BLAS, 不把更换编译选项混入量化对比。
- 本周以源码阅读和已有工具为主, 无需增加 backend、量化格式或修改 kernel。学习记录继续放在 `/home/king/llama-study/`, 遵守 [AGENTS.md](../../AGENTS.md) 与 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
- 必做量化知识覆盖 F16、Q8_0、Q4_0、Q4_K 和 Q4_K_M; 默认只生成一份 Q4_K_M。完整 Q8_0 benchmark、imatrix 校准、CUDA timeline、长上下文和 KV 量化均为选做。
- 本文是待执行的学习任务, 不代表 benchmark、数值测试或质量验收已经通过。所有结果表先留空, 只填实际测得的数据。

开始前闭卷回答: 一个 prompt 为什么会被切成多个 ubatch? graph reuse 复用了什么? 权重、KV Cache 和计算中间结果各活多久? 若回答不清, 先回看第 3、4、5 周相应图示, 不重新阅读全部代码。

## 7 天安排

| 天 | 时间 | 核心问题 | 当天主要输出 |
| --- | ---: | --- | --- |
| Day 1 | 2h | tensor 的 shape、stride、数据和 op 怎样共同表达计算? | tensor 记录表、MUL_MAT shape 图 |
| Day 2 | 2h | registry、device、backend、buffer type、buffer 各负责什么? | 对象关系图、设备与构建记录 |
| Day 3 | 2h | scheduler 如何分配节点、切图、分配内存和搬运数据? | split 图、一次实际 decode 的断点记录 |
| Day 4 | 2h | 同一个 MUL_MAT 怎样选择 CPU 执行路径? | CPU 调用链、PP/TG shape 对照 |
| Day 5 | 2h | block quantization 节省了什么, 又增加了什么工作? | block 大小手算、逐 tensor 量化决策表 |
| Day 6 | 4h | 怎样得到可复核的 PP/TG 与内存对照? | F16/Q4_K_M 基线、原始 JSON 与资源日志 |
| Day 7 | 4h | 数据能支持什么结论, 还缺什么证据? | 数值测试、质量观察、性能报告与口述验收 |

## 统一术语与观察表

| 对象 | 本周含义 | 需要避免的混淆 |
| --- | --- | --- |
| `ggml_context` | 管理 GGML 对象的内存 arena; 是否分配 tensor 数据取决于 `no_alloc` 等配置 | 不等于持有推理状态的 `llama_context` |
| `ggml_tensor` | 同时描述 type、shape、stride、数据位置及生成它的 op 和来源 | metadata 在 host 不表示 data 也在 host |
| `ggml_cgraph` | 节点、叶子和依赖关系组成的计算图 | 构图不等于执行; graph node 不一定对应一次 kernel launch |
| registry | backend 实现的注册入口, 可枚举其设备 | registry 不等于一张 GPU |
| device | 可查询能力、内存和默认 buffer type 的设备抽象 | 发现设备不等于所有 op 都能在该设备运行 |
| backend | 从设备初始化的执行实例, 提供计算、传输和同步接口 | backend handle 不等于权重 buffer |
| buffer type | 分配规则、alignment、设备归属和访问能力 | 一种 buffer type 可以创建多个 buffer |
| buffer | 一次实际分配或对既有存储的包装, 可以容纳多个 tensor | tensor 的逻辑字节数不等于 buffer 的实际分配量 |
| scheduler | 根据放置、能力和优先级分配节点, 切图并协调复制与执行 | 不保证全局最优调度, 也不保证异步调用一定重叠执行 |
| allocator | 根据图依赖与存活区间安排数据空间并复用可复用的存储 | 计算 buffer 大小不是所有中间 tensor 大小的简单求和 |

每天至少跟踪一个权重、一个中间 tensor 和一个输出, 记录:

```text
phase, tensor_name, op, src_names, type, ne[4], nb[4], view_src,
logical_bytes, buffer_type, assigned_backend, lifetime, evidence
```

AI 编译器经验可用于理解 graph IR、placement、capability query、memory planning 和 runtime dispatch。先在 GGML 中找到对应对象, 再判断哪些行为相似; 不预设它采用了与你熟悉的编译器相同的 lowering 或优化流程。

## Day 1: Tensor、type traits、op 和 cgraph

**投入: 2 小时。** 把模型语义落实到 GGML 对象和内存布局, 不读全部 op。

### 学习路线

1. **0-10 分钟**: 从第 3 周的单层图选出 Q projection, 写出输入 hidden state、权重和输出的语义 shape。
2. **10-35 分钟**: 阅读 [`ggml/include/ggml.h`](../../ggml/include/ggml.h) 的 `ggml_init_params`、`ggml_tensor`、`ggml_type_traits` 及 `ggml_mul_mat()` 声明。标出 `ne`、`nb`、`op`、`src`、`view_src`、`view_offs`、`buffer` 和 `data`。
3. **35-60 分钟**: 阅读 [`ggml/src/ggml.c`](../../ggml/src/ggml.c) 的 type traits、`ggml_row_size()`、`ggml_nbytes()`、`ggml_mul_mat()` 与 `ggml_build_forward_expand()`。区分创建结果 tensor、记录依赖和运行计算。
4. **60-85 分钟**: 阅读 [`ggml/src/ggml-impl.h`](../../ggml/src/ggml-impl.h) 的 `ggml_cgraph` 定义。对照已有 graph 记录, 找到 nodes/leafs、一个 view 和一个有计算的 op。
5. **85-105 分钟**: 完成下方 shape 与字节数手算, 从本地模型选一个实际 projection 核对。尚未拿到断点记录的字段留到 Day 3 补齐。
6. **105-120 分钟**: 画出 `tensor object -> buffer -> storage`, 说明为什么 `no_alloc=true` 时仍能创建 tensor 和 graph。

### 一个小型 MUL_MAT 推演

先固定二维、连续布局, 不加入 broadcast 和 view:

```text
W.ne = [K, M, 1, 1]
X.ne = [K, N, 1, 1]
Y = ggml_mul_mat(ctx, W, X)
Y.ne = [M, N, 1, 1]

Y[m, n] = sum_k W[k, m] * X[k, n]
```

`ne[0]` 是 GGML 的第 0 维, 不要直接照搬数学书中行、列的书写顺序。取 `K=4,M=3,N=2`, 手算一个输出元素; 将 `N` 改成 1, 解释这怎样对应单 sequence decode 的投影形态。

对 F16 的 `W`, 连续布局下 `nb[0]=2`、`nb[1]=K*2`。对 block quantized tensor, `ggml_type_size()` 表示一个 block 的存储字节数, 不是一个标量的字节数; `nb[0]` 不能用于把每个量化值当普通标量寻址。

```text
row_bytes = K / block_size * type_size
tensor_bytes = row_bytes * M
```

这个式子要求 `K` 满足 block 对齐, 且 tensor 连续、没有额外布局开销。view、padding、backend 专用布局和实际 allocation 必须另外检查。

### 当天输出与验收

- `notes/week7-day1.md`: Q projection 的语义 shape 与 GGML `ne/nb` 对照, 至少一个 view 的来源。
- `results/week7/tensors.csv`: 权重、中间结果、输出各一行, 未实测字段注明待核对。
- 能解释 `ggml_mul_mat()` 只建立计算关系, 以及 `sizeof(ggml_tensor)`、tensor 数据大小、buffer 分配大小为什么不同。

## Day 2: Backend 对象、设备能力与实验环境

**投入: 2 小时。** 用真实设备和对象验证接口边界, 准备 Debug/Release 工具。

### 学习路线

1. **0-10 分钟**: 闭卷画出昨天的 tensor/data 关系。
2. **10-40 分钟**: 阅读 [`ggml/include/ggml-backend.h`](../../ggml/include/ggml-backend.h) 的 device、buffer、compute、event 接口, 对照 [`ggml/src/ggml-backend-impl.h`](../../ggml/src/ggml-backend-impl.h) 的 interface 函数表。
3. **40-60 分钟**: 阅读 [`ggml/src/ggml-backend-reg.cpp`](../../ggml/src/ggml-backend-reg.cpp) 的注册、枚举和 `ggml_backend_load_all()`; 回看 [`src/llama-context.cpp`](../../src/llama-context.cpp) 的 backend 初始化。画出 registry -> device -> backend, 另画 device -> buffer type -> buffer。
4. **60-85 分钟**: 配置或增量构建下方工具, 保存硬件、版本、帮助信息和设备日志。构建等待期间继续阅读接口。
5. **85-105 分钟**: 核对 `supports_op`、`supports_buft` 和 `offload_op` 的区别; 有 CUDA 环境时执行下方诊断。
6. **105-120 分钟**: 填写设备能力表, 决定周末是否执行 CUDA 对照。

### 统一环境与构建

```bash
cd /home/king/llama.cpp
export MODEL="$PWD/deepseek-r1-1.5b-f16.gguf"
export STUDY_DIR=/home/king/llama-study
export WEEK7_RESULTS="$STUDY_DIR/results/week7"
export WEEK7_THREADS=4
mkdir -p "$STUDY_DIR/notes" "$STUDY_DIR/experiments" "$WEEK7_RESULTS"
git rev-parse HEAD > "$WEEK7_RESULTS/commit.txt"
git diff --stat > "$WEEK7_RESULTS/worktree-diff-stat.txt"
lscpu > "$WEEK7_RESULTS/lscpu.txt"
free -h > "$WEEK7_RESULTS/memory.txt"
stat -c '%n %s bytes' "$MODEL" > "$WEEK7_RESULTS/model-size.txt"

cmake -S . -B build-study-debug \
  -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=OFF -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build-study-debug --target \
  llama-simple llama-bench test-backend-ops test-quantize-fns -j 4

cmake -S . -B build-study-release \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build-study-release --target \
  llama-bench llama-quantize llama-completion llama-perplexity \
  test-backend-ops test-quantize-fns -j 4

cp build-study-debug/CMakeCache.txt "$WEEK7_RESULTS/cmake-debug.txt"
cp build-study-release/CMakeCache.txt "$WEEK7_RESULTS/cmake-release.txt"
./build-study-release/bin/llama-bench --help > "$WEEK7_RESULTS/bench-help.txt" 2>&1
./build-study-release/bin/llama-quantize --help > "$WEEK7_RESULTS/quantize-help.txt" 2>&1
./build-study-release/bin/llama-bench --list-devices > "$WEEK7_RESULTS/devices.txt" 2>&1
```

`WEEK7_THREADS=4` 是可复现的起点, 不代表机器最优线程数; 不足 4 核时减小, 异构核机器另外记录核类型和绑定策略。已有 CMake cache 若启用了 Vulkan、SYCL 等其他 backend, 应显式关闭或另建干净的 CPU 构建目录, 并核对运行日志。

某些工具打印帮助后返回非零退出码, 要结合输出判断。`--list-devices` 面向可用的 offload 设备, 没列出 CPU 不等于 CPU backend 缺失; 用注册日志或 `test-backend-ops ... -b CPU` 补充验证。构建耗时超过预算时减少选做任务, 不使用 Debug 数据代替 Release 基线。

### CUDA 诊断与分支选择

仅在有 NVIDIA GPU 时运行并保存结果:

```bash
nvidia-smi > "$WEEK7_RESULTS/nvidia-smi.txt" 2>&1
nvcc --version > "$WEEK7_RESULTS/nvcc-version.txt" 2>&1
```

分别记录 GPU 型号、driver、实际 Toolkit、CMake 选用的编译器和 backend 加载结果。`nvidia-smi` 显示的 CUDA 版本不能代替实际 Toolkit 版本。按本地 [构建文档](../build.md) 核对环境, 最后用短推理确认设备能执行; 仅有 `libggml-cuda.so` 不构成运行成功的证据。

设备能力表至少包括: `device name/type/default buft/host buffer/async/events/MUL_MAT support/evidence`。`supports_op` 需要具体的 op、shape 和 type, 不把一次支持结果推广到整个算子名下的所有配置。

### 当天输出与验收

- `notes/week7-day2.md`: backend 对象关系图、设备能力表、CUDA 可用或跳过的原因。
- `results/week7/` 中有构建、模型、硬件和工具参数记录。
- 能解释“buffer 在哪里”“哪个 backend 执行”“是否需要 copy”是三个相关但不同的问题。

## Day 3: Scheduler、分图、内存规划与跨设备复制

**投入: 2 小时。** 跟踪一次实际 decode, 区分 placement、allocation 和 execution。

### 学习路线

1. **0-10 分钟**: 回忆第 4 周 `process_ubatch()` 到 `graph_compute()` 的路径。
2. **10-40 分钟**: 阅读 [`ggml/src/ggml-backend.cpp`](../../ggml/src/ggml-backend.cpp) 的 `ggml_backend_sched_split_graph()`。关注预分配 tensor、权重位置、backend 优先级、op/buffer 支持和 view 的处理。
3. **40-60 分钟**: 阅读 `ggml_backend_sched_alloc_graph()`、`ggml_backend_sched_alloc_splits()`; 对照 [`ggml/include/ggml-alloc.h`](../../ggml/include/ggml-alloc.h) 与 [`ggml/src/ggml-alloc.c`](../../ggml/src/ggml-alloc.c) 的 gallocr reserve/alloc。只追踪一条路径, 标出存活区间和空间复用。
4. **60-80 分钟**: 阅读 `ggml_backend_sched_graph_compute_async()`、`ggml_backend_sched_compute_splits()`。标出 split input、复制、event 等待、backend compute 和同步。
5. **80-105 分钟**: 用 GDB 观察一个 prompt 和后续单 token decode, 记录 split 数、backend、输入和 graph 节点数。
6. **105-120 分钟**: 输出实际 split 图; 若只有 CPU, 再画一张明确标注为源码推演的 CPU/GPU copy 图。

### 最小断点流程

```bash
gdb --args ./build-study-debug/bin/llama-simple \
  -m "$MODEL" -ngl 0 -n 4 'Explain memory bandwidth briefly.'
```

```gdb
set breakpoint pending on
break llama_context::graph_compute
run
bt
break ggml_backend_sched_compute_splits
continue
print sched->n_backends
print sched->n_splits
print sched->splits[0].backend_id
print sched->splits[0].i_start
print sched->splits[0].i_end
print sched->splits[0].n_inputs
print sched->splits[0].graph.n_nodes
```

在当前 split 中选一个 `MUL_MAT` 节点, 查看其 `src`、type、`ne/nb`、buffer 和 view 信息。需要观察 split/alloc 时另开一次运行, 分别添加 `ggml_backend_sched_split_graph`、`ggml_backend_sched_alloc_splits` 断点, 不一次打开全部断点。

构造 context 时可能用测量图做 reserve; 将这些命中标记为 `reserve`, 与真正的 `prefill/decode` 分开。graph reuse 时也不要求每个 token 都重新 split/alloc。可单独用 `GGML_SCHED_DEBUG=1` 辅助查看 assignment, 但日志级别需允许 DEBUG, 且性能测量时应关闭。

### 跨设备图应说明什么

```text
producer on backend A
  -> source tensor in buffer A
  -> split input copy / compatible buffer access
  -> tensor consumed by backend B
  -> wait for required data, compute, record completion
```

不是所有 backend 边界都必然产生物理复制, 要核对目标 backend 是否能访问已有 buffer。也不是所有复制都异步: 当前 scheduler 对用户输入、event 和 fallback 同步有不同处理。函数名中含 `async` 不能作为 CPU/GPU 重叠执行的证据。

CPU-only 运行没有真实 host/device copy 也是有效结果, 不能为了填表声称观察到了 CUDA 传输。有可用 CUDA 时再用部分 offload 捕获真实边界, 记录一条具体的源/目标 tensor、字节数、复制入口和同步原因。

### 当天输出与验收

- `notes/week7-day3.md`: 真实 CPU split 图, 以及有证据标注的可选跨设备图。
- `results/week7/scheduler-trace.txt`: 区分 reserve、prefill、decode 的断点与调用栈。
- 能解释 compute buffer 为什么可以复用, 以及异步提交返回后为什么不能任意读取结果或复用输入。

## Day 4: CPU MUL_MAT 与可选 CUDA 分派

**投入: 2 小时。** 从 backend 接口走到一个具体计算路径, 学会解释性能变量来自哪里。

### 学习路线

1. **0-10 分钟**: 重画 Day 1 的 `W[K,M]`、`X[K,N]`、`Y[M,N]`, 对照多 token 和单 token 输入。
2. **10-35 分钟**: 阅读 [`ggml/src/ggml-cpu/ggml-cpu.cpp`](../../ggml/src/ggml-cpu/ggml-cpu.cpp) 的 `ggml_backend_cpu_graph_compute()`, 找出 plan、work buffer 与线程池。
3. **35-65 分钟**: 阅读 [`ggml/src/ggml-cpu/ggml-cpu.c`](../../ggml/src/ggml-cpu/ggml-cpu.c) 的 `ggml_graph_compute()`、`ggml_compute_forward()`、`ggml_compute_forward_mul_mat()` 与 `type_traits_cpu`。
4. **65-90 分钟**: 在下方小型 backend 测试中观察 `src0/src1/dst`、`vec_dot_type` 和工作区; 对照 [`ggml/src/ggml-cpu/quants.c`](../../ggml/src/ggml-cpu/quants.c) 与实际架构目录的 vec-dot 实现入口。
5. **90-105 分钟**: 有余力时阅读 CUDA 分派; 否则整理 CPU thread/chunk 与 type 路径。
6. **105-120 分钟**: 写出两条待验证的性能假设, 分别对应 PP 和 TG。

### 用小算子测试代替反复加载模型

当前测试集中有 `m=16,k=256,n=1/2` 的用例, 可先做能力查询:

```bash
./build-study-debug/bin/test-backend-ops support -b CPU -o MUL_MAT \
  -p 'type_a=(f16|q4_0),type_b=f32,m=16,n=(1|2),k=256,' \
  > "$WEEK7_RESULTS/mul-mat-support.txt" 2>&1

gdb --args ./build-study-debug/bin/test-backend-ops test -b CPU -o MUL_MAT \
  -p 'type_a=q4_0,type_b=f32,m=16,n=1,k=256,'
```

```gdb
set breakpoint pending on
break ggml_compute_forward_mul_mat
run
print dst->type
print dst->ne
print dst->src[0]->type
print dst->src[0]->ne
print dst->src[1]->type
print dst->src[1]->ne
print params->ith
print params->nth
bt
```

在局部变量初始化后再查看 `vec_dot_type`、`from_float` 和工作区。测试既运行被测 backend, 也运行 CPU reference, 用调用栈和 `params->use_ref` 区分。若命中了特殊优化入口, 记录实际分支, 不要求每种类型和 shape 都经过同一个 vec-dot 函数。

需要回答: Q4 权重与 F32 activation 怎样参与 dot product? 是否转换 activation 到另一种类型? 临时空间在哪里? 不要把量化推理理解为先将整个模型解量化为 F32 常驻内存。

### CUDA 只读到 launcher 边界

选读 [`ggml/src/ggml-cuda/ggml-cuda.cu`](../../ggml/src/ggml-cuda/ggml-cuda.cu) 的 `ggml_backend_cuda_graph_compute()`、`ggml_cuda_compute_forward()`、`ggml_cuda_mul_mat()`。记录 shape、dtype 和硬件条件怎样影响 mat-vec、mat-mat、量化路径或 cuBLAS 路径; 注意融合与 CUDA graph 可能改变实际调用形式。

本周无需读完 CUDA 模板和指令。没有 NVIDIA GPU 时将此项写成源码阅读记录, 不声称已观察 kernel launch。

### 当天输出与验收

- `notes/week7-day4.md`: CPU 调用链, 一个实际 MUL_MAT 的输入/输出/类型/线程记录。
- 能提出有条件的假设: 多 token projection 可以复用权重, 可能提高计算利用率; 单 sequence TG 每次输入较少, 可能更受权重读取带宽影响。
- 能说明以上是待验证假设, 长上下文 attention、模型大小、cache、线程和 kernel 都可能改变瓶颈。

## Day 5: Block quantization、Q4_K_M 与 imatrix

**投入: 2 小时。** 理解存储、计算和质量三个维度, 不推导全部量化算法。

### 学习路线

1. **0-10 分钟**: 回忆 `type_size` 与 `block_size` 的区别。
2. **10-35 分钟**: 阅读 [`ggml/src/ggml-common.h`](../../ggml/src/ggml-common.h) 的 `block_q4_0`、`block_q8_0`、`block_q4_K`, 完成下方大小手算。
3. **35-60 分钟**: 阅读 [`ggml/src/ggml-quants.c`](../../ggml/src/ggml-quants.c) 的 `quantize_row_q4_0_ref()`、`dequantize_row_q4_0()` 与 `quantize_row_q8_0_ref()`。画出 scale、整数编码、重建值和误差的关系。
4. **60-85 分钟**: 阅读 [`tools/quantize/quantize.cpp`](../../tools/quantize/quantize.cpp) 与 [`src/llama-quant.cpp`](../../src/llama-quant.cpp), 跟踪 `llama_model_quantize()`、目标 tensor type 的选择和 `ggml_quantize_chunk()`。
5. **85-105 分钟**: 执行 dry-run, 选 5 个 tensor 记录量化决策。阅读 [`tools/imatrix/README.md`](../../tools/imatrix/README.md) 与 [`tools/imatrix/imatrix.cpp`](../../tools/imatrix/imatrix.cpp) 的统计入口, 找到校准数据如何进入量化过程。
6. **105-120 分钟**: 解释 Q4_K_M 的混合策略, 为周末选择默认实验或资源不足时的降级方案。

### Block 大小手算

下表为当前源码中的 block 存储, 不包含 GGUF metadata、文件 alignment 和 backend padding:

| type | 每 block 值数 | 每 block 字节 | 有效 bits/value | 需要解释 |
| --- | ---: | ---: | ---: | --- |
| F16 | 1 | 2 | 16 | 无 block scale |
| Q8_0 | 32 | 34 | 8.5 | 32 个 int8 和一个 FP16 scale |
| Q4_0 | 32 | 18 | 4.5 | 16 字节 packed quants 和一个 FP16 scale |
| Q4_K | 256 | 144 | 4.5 | super-block 内的 scales/mins 和 packed quants |

取 `K=256,M=16`, 分别算出上述四种连续权重 tensor 的逻辑大小。再选一行 Q4_0, 解释为什么读取一个值需要解码 nibble 与 scale, 而不是简单地做 `float *` 指针索引。

必须区分三个名字:

- `GGML_TYPE_Q4_K`: 单个 tensor 的元素存储类型。
- `Q4_K_M`: 整个模型的量化预设, 会按 tensor 角色和模型条件混合多种类型, 不能把整个文件都按 4 bit/weight 计算。
- `-ctk/-ctv`: context 的 K/V cache 类型, 不会因为权重被量化成 Q4_K_M 就自动变为 Q4。周末基线固定为 F16。

### 决策预览与 imatrix 的边界

```bash
./build-study-release/bin/llama-quantize --dry-run "$MODEL" Q4_K_M \
  > "$WEEK7_RESULTS/quantize-q4km-dry-run.txt" 2>&1
./build-study-release/bin/llama-quantize --dry-run "$MODEL" Q8_0 \
  > "$WEEK7_RESULTS/quantize-q8-dry-run.txt" 2>&1
df -h "$STUDY_DIR" > "$WEEK7_RESULTS/disk-space.txt"
```

选择 embedding、attention projection、FFN up/down、output 等共 5 个 tensor, 记录 `source type/target type/shape/estimated bytes/选择原因`。dry-run 只提供量化计划与大小估计, 不验证量化模型的运行速度、数值或文本质量。

imatrix 从校准数据的 activation 统计得到重要性信息, 可用于加权量化误差, 并在部分策略中影响类型选择。它不是重新训练模型, 也不保证任何数据集上都会提升质量。Q4_K_M 可以不提供 imatrix; 某些其他格式有不同要求, 应检查当前代码。

默认完成“无 imatrix 的 Q4_K_M”即可。选做时用有代表性的校准文本, 另外保留独立的质量评估文本, 记录数据来源、token 数、context、覆盖 tensor 和工具参数, 不用同一批文本校准又宣称泛化质量已验证。

### 当天输出与验收

- `notes/week7-day5.md`: block 大小推演、量化/解量化示意、imatrix 的用途与限制。
- `results/week7/quantization-decisions.csv`: 5 个真实 tensor 的决策记录。
- 能解释模型文件变小为什么不保证 PP 和 TG 都按相同比例变快。

## Day 6: Release benchmark 与内存证据

**投入: 4 小时。** 默认完成 CPU F16/Q4_K_M 对照, 先得到可信基线再扩展变量。

### 时间安排

| 时间 | 任务 |
| --- | --- |
| 0-20 分钟 | 检查环境、空间与短运行, 估算完整实验耗时 |
| 20-60 分钟 | 生成 Q4_K_M, 保存量化日志、模型大小与来源 |
| 60-140 分钟 | 顺序运行 4 个基线单元, 每个单元 warmup 后重复 3 次 |
| 140-185 分钟 | 记录内存分类, 增加一个线程数对照或选做 CUDA 对照 |
| 185-215 分钟 | 核对原始数据、波动和无效实验, 必要时补测 |
| 215-240 分钟 | 整理结果表, 写出待验证的瓶颈解释 |

### 1. 固定测试口径

先阅读 [`tools/llama-bench/README.md`](../../tools/llama-bench/README.md), 再核对 [`llama-bench.cpp`](../../tools/llama-bench/llama-bench.cpp) 的 `test_prompt()`、`test_gen()`、warmup 和计时边界。

| 指标/参数 | 本周口径 |
| --- | --- |
| PP | `-p 512 -n 0`, 测多 token prompt processing |
| TG | `-p 0 -n 128 -d 0`, 测从零 depth 开始的逐 token decode |
| `-p 512 -n 128` | 生成独立 PP 和 TG 测试项, 不是在同一个 512-token prompt 后测 TG |
| `-pg 512,128` | 联合计时 PP + TG, 不能将合并吞吐当作纯 TG |
| `-d 512` | 计时前准备指定深度的 KV; 可用于单独研究已有上下文下的 TG |
| context 容量 | 当前工具按 `n_prompt+n_gen+n_depth` 请求, runtime 可能调整或对齐; 保存实际日志 |
| 线程/batch/ubatch | 固定 `-t "$WEEK7_THREADS" -b 512 -ub 128` |
| cache/FA | 固定 `-ctk f16 -ctv f16 -fa off`, 避免 auto 改变对照条件 |
| 重复与 warmup | 保留默认 warmup, `-r 3`, 保存每次 `samples_ts` 和均值/标准差 |
| 计时范围 | 不含模型加载、tokenization 和 sampler; TG 使用测试 token, 不是自然语言质量测试 |

只在相同测试单元中比较 F16/Q4_K_M。PP 与 TG 请求的 context 容量不同, 不能用它们的 RSS 差值直接归因于阶段。`-d` 还可能引入 host state 保存 buffer, 研究深度时也要记录这部分开销。

先用 F16 跑 `-p 32 -n 8 -r 1` 的短测试估时, 该结果不进入正式表。若机器较慢, 将整个对照统一缩为 `pp128/tg32`, 保留每组 3 次重复, 明确改过的长度。不要只缩短其中一个模型的测试。

### 2. 生成一份量化模型

确认目标文件名尚未使用、磁盘与内存足够后执行。直接从原始 F16 权重量化, 不从另一份低 bit 模型再次量化:

```bash
export WEEK7_QMODEL="$STUDY_DIR/experiments/week7-q4_k_m.gguf"
./build-study-release/bin/llama-quantize \
  "$MODEL" "$WEEK7_QMODEL" Q4_K_M "$WEEK7_THREADS" \
  > "$WEEK7_RESULTS/quantize-q4km.txt" 2>&1
stat -c '%n %s bytes' "$MODEL" "$WEEK7_QMODEL" \
  > "$WEEK7_RESULTS/model-sizes.txt"
sha256sum "$MODEL" "$WEEK7_QMODEL" > "$WEEK7_RESULTS/model-sha256.txt"
```

量化后先验证加载和短推理, 再执行完整矩阵。资源不足时保留 dry-run, 改为同一 F16 模型的两档线程数对照, 完成 PP/TG、内存与测试主线; 在报告中明确量化速度和质量尚未验证, 不把估计大小当成实测结果。

### 3. 四个基线单元

每条命令只跑一个模型和一个阶段, 使 `/usr/bin/time` 的最大 RSS 能对应到该单元。正式测量时关闭 GDB、scheduler debug、后台构建及其他推理任务。

```bash
/usr/bin/time -v -o "$WEEK7_RESULTS/f16-pp.time.txt" \
  ./build-study-release/bin/llama-bench \
  -m "$MODEL" -ngl 0 -p 512 -n 0 -d 0 \
  -t "$WEEK7_THREADS" -b 512 -ub 128 -ctk f16 -ctv f16 -fa off -r 3 -o json \
  > "$WEEK7_RESULTS/f16-pp.json" 2> "$WEEK7_RESULTS/f16-pp.stderr.txt"

/usr/bin/time -v -o "$WEEK7_RESULTS/f16-tg.time.txt" \
  ./build-study-release/bin/llama-bench \
  -m "$MODEL" -ngl 0 -p 0 -n 128 -d 0 \
  -t "$WEEK7_THREADS" -b 512 -ub 128 -ctk f16 -ctv f16 -fa off -r 3 -o json \
  > "$WEEK7_RESULTS/f16-tg.json" 2> "$WEEK7_RESULTS/f16-tg.stderr.txt"

/usr/bin/time -v -o "$WEEK7_RESULTS/q4km-pp.time.txt" \
  ./build-study-release/bin/llama-bench \
  -m "$WEEK7_QMODEL" -ngl 0 -p 512 -n 0 -d 0 \
  -t "$WEEK7_THREADS" -b 512 -ub 128 -ctk f16 -ctv f16 -fa off -r 3 -o json \
  > "$WEEK7_RESULTS/q4km-pp.json" 2> "$WEEK7_RESULTS/q4km-pp.stderr.txt"

/usr/bin/time -v -o "$WEEK7_RESULTS/q4km-tg.time.txt" \
  ./build-study-release/bin/llama-bench \
  -m "$WEEK7_QMODEL" -ngl 0 -p 0 -n 128 -d 0 \
  -t "$WEEK7_THREADS" -b 512 -ub 128 -ctk f16 -ctv f16 -fa off -r 3 -o json \
  > "$WEEK7_RESULTS/q4km-tg.json" 2> "$WEEK7_RESULTS/q4km-tg.stderr.txt"
```

当前 `llama-bench` 默认可能屏蔽模型/context 的详细日志。内存归因需要日志时, 另加 `-v` 跑一次短观察, 不将带大量日志的耗时混入正式均值。`time -v` 的 RSS、page faults 和 wall time 覆盖进程全过程, 包括加载与 warmup; 其 wall time 不能替代 JSON 中的 PP/TG 时间。

### 4. 内存必须分账

| 类别 | 所有权/存活时间 | 记录来源与解释 |
| --- | --- | --- |
| 权重 | model, 通常跨多次 decode | 模型 tensor/buffer 日志; 文件大小不等于当前驻留内存 |
| KV/model memory | context, 跨 token 保留 | 第 5 周估算和实际 cache buffer 日志; 受容量、层数、head、type 等影响 |
| compute/临时工作区 | scheduler/backend, 可复用 | `sched_reserve()`、gallocr、CPU plan 和 compute buffer 日志 |
| 输出与 host 辅助区 | context/工具 | logits、embedding、batch、state 缓存及 staging 等 |
| RSS/PSS | 操作系统统计 | RSS 是驻留页, 可能包含映射文件和共享页; PSS 用于分摊共享页 |
| VRAM | GPU runtime/device | 权重、KV、compute、runtime 与其他占用; 不等于模型文件大小 |

从第 5 周的 KV 估算出发, 对单 sequence、普通 dense Transformer 的 F16 K/V 写出:

```text
KV bytes ~= n_layer * allocated_cells *
            (n_embd_k_gqa * 2 + n_embd_v_gqa * 2)
```

说明该式忽略了 alignment、padding 和实现开销, 使用分配容量而非当前已填 token 数; 不直接套到 recurrent、hybrid 或特殊 cache 布局。权重量化主要改变权重部分, 本组固定 F16 KV 后 cache 不会随权重 bit 数等比例缩小。

可在单独的长运行中读取 `/proc/<实际 PID>/smaps_rollup`, 标明采样时点。mmap 页会按访问驻留, 不把虚拟映射大小当物理内存, 不把所有类别与 RSS 再次相加。有 CUDA 时按进程采样 VRAM, 记录采样间隔, 将采样最大值标为“观测峰值”; 不能据此保证捕获了瞬时峰值。

### 5. 只增加一个扩展变量

CPU 路线优先选择同模型、同长度的两档线程数, 或只改变 `-ub` 观察 PP。不要在同一对照中同时改变线程、ubatch、FA 和量化类型。

CUDA 路线先按 Day 2 的构建配置另建 `build-study-cuda`, 将 build type 设为 Release、`GGML_CUDA=ON`、`GGML_BLAS=OFF`, 构建 `llama-bench` 和 `test-backend-ops`。使用同一个 CUDA 二进制分别测 CPU 与 offload, 其他参数保持一致。CPU 对照可用 `-dev none -ngl 0 -nopo 1`, 并核对日志无设备计算; GPU 对照用实际设备名、`-ngl 99 -nopo 0`。

`-ngl 99` 是请求, 需要日志确认实际 offload 层数与设备; 不能单凭命令行声称模型完全在 GPU 上。额外选择一个可容纳的部分 offload 层数时, 将其用于 Day 3 的传输证据, 不要求速度一定随 offload 层数单调增加。

### 当天输出与验收

- `results/week7/benchmark.csv`: 至少包含配置、model hash、test、线程、batch/ubatch、KV/FA、实际 context、重复数、均值、标准差、RSS 和原始文件路径。
- `notes/week7-day6.md`: 模型、KV、compute、output 的内存分账, 至少一项估算与实际日志的核对。
- 每组保留至少 3 个有效样本; 超时、OOM、后台负载等异常单独标记, 不静默删除不利结果。

## Day 7: 数值验证、质量观察与性能结论

**投入: 4 小时。** 把“看到差异”推进到“能说明证据支持到哪一步”。

### 时间安排

| 时间 | 任务 |
| --- | --- |
| 0-20 分钟 | 闭卷解释 tensor -> scheduler -> backend 主线, 检查原始数据 |
| 20-65 分钟 | 运行 quantization 与选定 MUL_MAT 数值测试 |
| 65-115 分钟 | 做固定输入质量观察; 有预算时补小型 perplexity 对照 |
| 115-160 分钟 | 选择一个性能差异, 做一次针对性的补充观测 |
| 160-215 分钟 | 写性能报告, 标明事实、假设和限制 |
| 215-240 分钟 | 完成口述验收, 整理交给第 8 周的配置与内存预算 |

### 1. 现有测试与证据范围

```bash
./build-study-release/bin/test-quantize-fns \
  > "$WEEK7_RESULTS/test-quantize-fns.txt" 2>&1

./build-study-release/bin/test-backend-ops test -b CPU -o MUL_MAT \
  -p 'type_a=(f16|q8_0|q4_0|q4_K),type_b=f32,m=16,n=(1|2),k=256,' \
  > "$WEEK7_RESULTS/test-mul-mat-cpu.txt" 2>&1
```

执行后记录退出码、实际执行用例数、跳过数和失败项。`test-backend-ops` 默认跳过 CPU, 所以 CPU 必须显式指定 `-b CPU`; 用 `-p` 缩小范围后必须确认有匹配用例, “零用例成功退出”不是通过。`support` 模式只报告能力, 不能代替 `test` 的数值比较。

当前 `test` 模式会初始化 CPU reference, 显式选择 CPU 时可比较被测 CPU 路径与 reference。若有 CUDA, 用 CUDA 构建和实际设备名如 `-b CUDA0` 执行同一组 case, 并记录不支持的类型/shape。这是所选算子的数值证据, 不是完整模型质量或全部 backend 的验证。

`test-quantize-fns` 检查量化、解量化和 dot product 等函数在合成数据上的误差。测试通过不能证明 Q4_K_M 在你的任务上质量不变, 性能计时也应来自 Release benchmark 而不是 Debug 测试耗时。

### 2. 质量观察必须独立于 benchmark

复用第 6 周的生成程序或已验证的 `llama-completion` 配置, 预先固定 3 个短输入: 一个事实问答、一个简单推理、一个 Planner JSON。两个模型使用同样的实际 prompt/template、greedy sampler、token 上限、context 和 grammar 设置。

记录输出 token 数、停止原因、完整输出、格式是否合法和任务是否答对。若模型输出含 reasoning 段, 沿用第 6 周确定的最终 JSON 边界, 不把整个输出直接当 JSON。第 6 周能生成 JSON 不表示量化版本自动继承该验收结果。

- 同样条件下输出文本不同不自动表示量化实现有 bug; 权重误差可能改变 token 选择并逐步放大。
- 相同 seed 不能抵消权重或 backend 数值差异; greedy 也不保证跨精度逐 token 一致。
- 三个输入只能作质量冒烟检查, 不能据此得出总体质量无损或提升的结论。
- 已有实现可导出固定前缀下的 logits 时, 选做比较同一前缀的 logits/概率差异; 不比较两条已经分叉的生成历史并归因于单步量化误差。

选做困惑度对照时阅读 [`tools/perplexity/README.md`](../../tools/perplexity/README.md), 准备足够长度且未用于 imatrix 校准的本地语料, 先核对 `llama-perplexity --help`。示例配置如下, 两个模型只替换 `-m` 与输出文件名:

```bash
./build-study-release/bin/llama-perplexity \
  -m "$MODEL" -f "$STUDY_DIR/experiments/week7-eval.txt" \
  -ngl 0 -c 512 -b 512 -ub 128 -t "$WEEK7_THREADS" \
  -ctk f16 -ctv f16 -fa off --chunks 2 \
  > "$WEEK7_RESULTS/ppl-f16.txt" 2>&1

./build-study-release/bin/llama-perplexity \
  -m "$WEEK7_QMODEL" -f "$STUDY_DIR/experiments/week7-eval.txt" \
  -ngl 0 -c 512 -b 512 -ub 128 -t "$WEEK7_THREADS" \
  -ctk f16 -ctv f16 -fa off --chunks 2 \
  > "$WEEK7_RESULTS/ppl-q4km.txt" 2>&1
```

保存语料 hash、实际 token/chunk 数和困惑度差异。两个 chunk 只是小样本演示, 不能替代代表性评估集; 不把不同 tokenizer、不同语料或不同 context 的 perplexity 直接比较。没有实际量化模型时本节量化对照明确标记未完成。

### 3. 从指标到瓶颈假设

先用简单关系组织判断:

```text
arithmetic_intensity = useful_operations / bytes_transferred
time_lower_bound ~= max(operations / compute_rate, bytes / bandwidth)
measured_time also includes launch, synchronization and runtime overhead
```

模型文件字节数不等于实际 DRAM/PCIe 流量, FLOPs 也不描述全部解码、转换和量化 dot 工作。没有硬件计数器或 timeline 时, 把上述模型当解释框架, 不据此填写虚构带宽。

| 观察 | 可以提出的假设 | 下一条证据 |
| --- | --- | --- |
| Q4_K_M 的 TG 更快, PP 改善较小或更慢 | 减少权重读取的收益与量化计算开销不同 | CPU type/vec-dot 路径、线程响应、实际 profile |
| 增加线程没有提速 | 带宽饱和、调度开销、异构核或频率变化 | 固定其余配置的线程扫描与 CPU 状态 |
| GPU 部分 offload 比预期慢 | copy、同步或 CPU 剩余计算较重 | 真实 split 边界与 CUDA timeline |
| 增大 context/depth 后 RSS 和 TG 延迟上升 | KV 分配/访问、attention 和 state 辅助区增加 | 同容量/不同已用深度的独立实验、buffer 分账 |
| 第一轮明显慢或方差很大 | 页驻留、频率、温度或后台任务影响 | warmup 日志、page faults、重新安排实验顺序 |

可选 Linux 进程级观测, 事件与输出参数见 [perf-stat 手册](https://man7.org/linux/man-pages/man1/perf-stat.1.html):

```bash
perf stat -e task-clock,context-switches,page-faults \
  -o "$WEEK7_RESULTS/perf-tg.txt" -- \
  ./build-study-release/bin/llama-bench \
  -m "$MODEL" -ngl 0 -p 0 -n 128 -d 0 \
  -t "$WEEK7_THREADS" -b 512 -ub 128 -ctk f16 -ctv f16 -fa off -r 3 -o json \
  > "$WEEK7_RESULTS/perf-tg.json" 2> "$WEEK7_RESULTS/perf-tg.stderr.txt"
```

这条命令统计整个进程, 包含加载与 warmup, 不能把所有事件归到纯 TG。权限或工具不可用时保留原因, 用已获得的基线完成有限结论。硬件事件需先用本机 `perf list` 确认; cache misses 本身也不能直接证明内存带宽已饱和。CUDA 选做只需选取一个稳态时间窗口, 标出 kernel、copy 与等待, 不在本周优化 kernel。

### 4. 周报告模板

报告写入 `notes/week7-review.md`, 内容控制在能独立讲清的范围:

1. **问题与条件**: 本次比较什么, 固定了什么, commit/build/hardware/model hash 是什么。
2. **执行路径**: 一张实际 backend split 图, 一个 MUL_MAT 的 shape/type/执行入口, 未实测的路径明确标注。
3. **结果**: PP/TG 各 3 次样本、均值/标准差、模型大小、最大 RSS、可选 VRAM, 链接原始文件。
4. **内存解释**: weights/KV/compute/output 的分账, 估算与日志差异, 统计口径限制。
5. **正确性与质量**: 实际数值测试覆盖, 文本/JSON 观察, 可选 perplexity, 不混为一个结论。
6. **判断与下一步**: 一个有证据的性能差异, 最可能的原因, 仍存在的替代解释, 下一项单变量实验。
7. **第 8 周输入**: 选定模型、线程、context、batch/ubatch、backend 与资源预算, 说明选择条件。

若速度差与波动同量级, 可以得出“当前数据不足以区分”的结论。若只有 CPU 路线, 完整 CPU/量化或 CPU/线程对照即可验收, 不要求编造 GPU 结果。

### 第 7 周验收清单

- [ ] 能从一个 projection 的语义 shape 写出 GGML `ne/nb`, 区分 metadata、data 和 buffer。
- [ ] 能画出 registry/device/backend/buffer type/buffer/scheduler 的关系。
- [ ] 能用一次实际 decode 说明 assignment、split、allocation、copy 与 synchronization 的位置。
- [ ] 能区分 reserve 图与实际推理图, 说明 graph reuse 不等于重新分配全部计算内存。
- [ ] 能追到 CPU MUL_MAT 的 type/vec-dot/工作区边界, 不将整个模型解量化当成必经过程。
- [ ] 能计算 Q8_0、Q4_0、Q4_K block 大小, 区分 Q4_K_M 预设和 KV 量化。
- [ ] 有至少一组 Release 对照, 每个 PP/TG 单元 warmup 后至少 3 次, 原始记录齐全。
- [ ] 能区分文件大小、buffer allocation、RSS/PSS、VRAM, 说明 mmap 与实际驻留的关系。
- [ ] 数值测试有实际用例与结果, 能说明算子测试、质量观察和性能测试各自的证据范围。
- [ ] 能用数据解释一个性能差异或说明证据不足, 并提出下一项可证伪的假设。
- [ ] CUDA 未完成时写明原因; 完成时有实际 offload、数据移动与设备内存证据。

## 本周完成包

```text
/home/king/llama-study/
  notes/
    week7-day1.md ... week7-day7.md
    week7-review.md
  experiments/
    week7-q4_k_m.gguf                  # if resources permit
    week7-eval.txt                     # optional quality corpus
  results/week7/
    commit.txt / worktree-diff-stat.txt
    cmake-debug.txt / cmake-release.txt
    lscpu.txt / memory.txt / disk-space.txt / devices.txt
    bench-help.txt / quantize-help.txt
    tensors.csv / scheduler-trace.txt / mul-mat-support.txt
    quantization-decisions.csv
    quantize-q4km-dry-run.txt / quantize-q8-dry-run.txt
    quantize-q4km.txt / model-sizes.txt / model-sha256.txt
    f16-pp.json / f16-tg.json / q4km-pp.json / q4km-tg.json
    *.stderr.txt / *.time.txt
    benchmark.csv / memory-breakdown.csv
    test-quantize-fns.txt / test-mul-mat-cpu.txt
    quality-observations.md
    optional-cuda-or-profile-records/
```

交给第 8 周的核心资产是可复现的运行配置、PP/TG 基线和每 context 的资源预算。共享一个 `llama_model` 可以复用模型权重, 多个 `llama_context` 仍会各自持有 KV、compute 和输出等状态; 用本周数据评估多 Agent 方案的容量, 第 8 周再讨论 Server 调度和端到端延迟。
