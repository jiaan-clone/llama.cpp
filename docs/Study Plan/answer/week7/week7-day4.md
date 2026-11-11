# week7 day4: CPU MUL_MAT 与可选 CUDA 分派

本文回答[第 7 周学习路线](../../007-llama-cpp-week7-daily-learning-route.md)中 Day 4 的当天输出与验收。

源码基准为 `697b4bbf4633a38abfd66f7cc54a167095aedef3`。本次只做源码阅读和 shape/字节数推导, 按要求跳过算子 support/test、GDB、模型推理及性能测试。实际执行分支、线程数和数值误差均未测量。

## 1. CPU 调用链

```text
llama_context::graph_compute()
  -> 设置 n_threads / threadpool
  -> ggml_backend_sched_graph_compute_async()
  -> ggml_backend_sched_compute_splits()
  -> ggml_backend_graph_compute_async(CPU, split_graph)
  -> CPU iface.graph_compute
  -> ggml_backend_cpu_graph_compute()
       -> ggml_graph_plan(cgraph, n_threads, threadpool)
       -> 准备/扩容 cpu_ctx->work_data
       -> ggml_graph_compute(cgraph, cplan)
            -> 线程池执行 ggml_graph_compute_thread()
            -> ggml_compute_forward(params, node)
                 -> extra buffer 优化分支, 或普通 op switch
                 -> GGML_OP_MUL_MAT
                 -> ggml_compute_forward_mul_mat()
                      -> 特殊 hint / 可用优化分支
                      -> 通用路径: 必要时转换 src1 到 vec_dot_type
                      -> barrier, 分配 output chunks
                      -> ggml_compute_forward_mul_mat_one_chunk()
                      -> type_traits_cpu[src0->type].vec_dot(...)
```

入口见 [ggml-cpu.cpp](../../../../ggml/src/ggml-cpu/ggml-cpu.cpp), plan、线程与通用 MUL_MAT 见 [ggml-cpu.c](../../../../ggml/src/ggml-cpu/ggml-cpu.c)。这条链描述可能进入通用 dot 的路径; extra buffer、架构优化和启用的 llamafile SGEMM 等可以提前接手, 不能声称所有 MUL_MAT 都逐一经过链尾。

`ggml_backend_cpu_graph_compute()` 用 `ggml_graph_plan()` 确定工作区和线程需求, 在容量不足时扩容 `cpu_ctx->work_data`, 将其交给 `cplan.work_data`。线程池存在时复用; 没有提供线程池时, `ggml_graph_compute()` 可以创建临时线程池。配置线程数、实际线程池容量和 OpenMP 实际线程数不一定相同, 最终要看 `params->nth`。

## 2. 一个 MUL_MAT 的静态记录与待实测项

沿用 Day 1 的第 0 层 Q projection, 无 LoRA、普通独立 Q 投影、连续布局:

| 字段 | 静态事实或推导 | 运行记录 |
| --- | --- | --- |
| `src0` | `blk.0.attn_q.weight`, F16, `[1536,1536,1,1]` | 文件 type/shape 已核对; 实际 buffer/nb 待查 |
| `src1` | norm 后 activation, F32, `[1536,N,1,1]` | 待断点确认 |
| `dst` | bias 前 Q, F32, `[1536,N,1,1]` | 待断点确认, 不与 bias/reshape/RoPE 同名节点混淆 |
| `vec_dot_type` | 通用 F16 权重路径为 F16 | 实际是否进入此路径待查 |
| `from_float` | 来自 F16 CPU trait, 将 F32 activation 转成 F16 | 函数指针和地址待查 |
| `params->ith/nth` | ith 是线程编号, nth 是此次计算的线程数 | 未实测, 不填假定的 4 |
| `params->wdata/wsize` | kernel 临时工作区 | 指针、完整大小待查 |
| `params->use_ref` 与调用栈 | 区分被测 backend 和 CPU reference | 未实测 |

路线中的小算子用例在 [test-backend-ops.cpp](../../../../tests/test-backend-ops.cpp) 中存在: `m=16,k=256,n=1/2`, W 取 F16 或 Q4_0, X 取 F32。它们适合观察单个算子, 不需要反复加载完整模型。`support` 只查询能力; `test` 才进入测试和 reference 比较, 即使输出成功也只覆盖命中的用例。

本次实际 MUL_MAT 的输入/输出/线程断点记录暂缺, 原因是跳过手动实验。下面的具体数值是供后续检查用的预期, 不能代替记录。

## 3. Q4 权重与 F32 activation 怎样参与 dot

通用 CPU `type_traits_cpu` 记录的部分组合如下:

| 权重 `src0->type` | dot 要求的 activation 类型 | dot 入口 | 原始 F32 activation 的处理 |
| --- | --- | --- | --- |
| F32 | F32 | `ggml_vec_dot_f32` | 通常直接使用 |
| F16 | F16 | `ggml_vec_dot_f16` | 转为 F16 临时数据 |
| Q4_0 | Q8_0 | `ggml_vec_dot_q4_0_q8_0` | 按 block 量化为 Q8_0 临时数据 |
| Q8_0 | Q8_0 | `ggml_vec_dot_q8_0_q8_0` | 按 block 量化为 Q8_0 临时数据 |
| Q4_K | Q8_K | `ggml_vec_dot_q4_K_q8_K` | 按 super-block 量化为 Q8_K 临时数据 |

关键点是 `from_float` 取自 `type_traits_cpu[vec_dot_type]`, 不是直接取权重 type 的转换器。对 Q4_0 权重, 被转换的是右侧 activation, 转换为 Q8_0, 不是将整个模型权重解量化成 F32。

```text
F32 X -> quantize_row_q8_0 -> temporary Q8_0 X
Q4_0 W + Q8_0 X -> block dot + scales -> F32 Y
```

[ggml-cpu/quants.c](../../../../ggml/src/ggml-cpu/quants.c) 的 `ggml_vec_dot_q4_0_q8_0_generic()` 将 Q4 nibble 解码为 `q-8`, 与 Q8 int8 相乘求和, 再乘两侧 block scale 并累加成浮点输出。可写为:

```text
dot(W,X) ~= sum_blocks dW*dX * sum_j ((qW[j]-8)*qX[j])
```

架构实现可以用 SIMD 完成同类工作。比如 x86 入口在 [arch/x86/quants.c](../../../../ggml/src/ggml-cpu/arch/x86/quants.c); 本次未采集硬件或实际二进制的指令配置, 不声称已执行某个 AVX/ARM kernel。

临时 activation 存在 `params->wdata`, 由 CPU plan/work buffer 提供。对于 `K=256,M=16` 的连续二维小用例:

| 通用路径 | N=1 的 activation 转换数据 | N=2 的 activation 转换数据 |
| --- | ---: | ---: |
| F16 权重, X 转 F16 | 512 bytes | 1024 bytes |
| Q4_0 权重, X 转 Q8_0 | `(256/32)*34=272` bytes | 544 bytes |

这是该节点转换数据的体积, 不是整个 graph 的 `work_size`: plan 还要覆盖其他节点的最大工作需求及实现的对齐/线程附加空间。图中 X 的 metadata 仍是 F32, 转换数据并不要求把原 tensor 改成 Q8_0。

## 4. 线程如何分配工作

普通路径先按 block 范围将 activation 转换工作分给线程, 再经 barrier 保证 dot 使用的临时数据已经就绪。结果空间按 `nr0=dst.ne[0]` 和 `nr1=dst.ne[1]*dst.ne[2]*dst.ne[3]` 分块。

当前初始 chunk size 通常为 16, 如果任一结果维度退化为 1 则用 64; 当 chunk 数相对线程太少或处于 NUMA 配置时, 改为按线程切分较大的维度。各线程先领取与 ith 对应的 chunk, 后续通过原子计数领取剩余工作。某些 dot kernel 可以一次处理多行, shape 不满足时退回单行处理。

因此提高线程数并不保证线性加速: 小矩阵可能没有足够工作, barrier、调度、缓存竞争及跨 NUMA 访问都可能影响结果。路线固定 4 线程是实验参数, 不是已经观察到每个算子都使用 4 个有效工作线程。

## 5. PP 与 TG shape 对照及两个可验证假设

| 条件 | W | X | Y | 潜在影响 |
| --- | --- | --- | --- | --- |
| PP, 当前 ubatch 含 N 个 token | `[K,M]` | `[K,N]`, N 通常大于 1 | `[M,N]` | 同一权重可服务多个 token, 为分块与向量化提供更多工作 |
| 单 sequence TG, 一个新 token | `[K,M]` | `[K,1]` | `[M,1]` | 每步仍要使用权重, 权重读取相对有用计算可能更突出 |

这里 N 是实际 ubatch 的 token 数, 不是直接等于命令行总 prompt 长度。多 sequence TG 也可以形成 N>1。

**假设一, PP:** 固定模型、CPU 编译选项、线程、KV 类型与 context, 在一定范围增加 ubatch token 数, 可能通过复用权重和更合适的矩阵计算路径提高每 token 吞吐。验证时固定总 prompt token 数, 只调整 ubatch, 比较 warmup 后重复样本, 同时记录实际 shape 和执行入口。若吞吐不升或下降, 应检查 cache 容量、工作区、attention 占比和 kernel 分支, 不能继续宣称“batch 越大越快”。

**假设二, TG:** 在模型权重无法有效常驻缓存、上下文较短且权重读取占比较高时, Q4_K_M 的单 sequence TG 可能因减少权重流量而快于 F16。验证时固定输入、线程、已用 KV 深度、batch/ubatch 和 KV 类型, 只换同源量化权重, 保留多次 Release 结果。若收益很小或变慢, 量化 activation、dot 实现、同步与其他算子开销都是替代解释。

以上均为待验证假设。长上下文 attention/KV 访问、模型大小、CPU cache、线程数、频率、NUMA、backend 和 kernel 都可能改变瓶颈; 没有 profile 或计数器不能把瓶颈确定为带宽。单个 projection 的权重复用也不能直接解释整个模型端到端延迟。

## 6. CUDA 分派的源码边界

[ggml-cuda.cu](../../../../ggml/src/ggml-cuda/ggml-cuda.cu) 中可从 `ggml_backend_cuda_graph_compute()` 的图执行走到普通节点的 `ggml_cuda_compute_forward()`, 在 `GGML_OP_MUL_MAT` 分支进入 `ggml_cuda_mul_mat()`。当前 launcher 按以下条件选择路径:

- hint、view 和 allocation padding 安全性, 以及 `src1/dst` type 会影响是否直接使用 cuBLAS 等分支。
- `ggml_cuda_should_use_mmvf/mmf` 判断浮点 mat-vec 或 mat-mat 路径。
- `ggml_cuda_should_use_mmvq/mmq` 判断量化 mat-vec 或 mat-mat 路径。
- 选择依赖权重 type、shape/stride、token 数、GPU compute capability 和 warp 等条件; 不满足时回退 cuBLAS。

融合与 CUDA graph 可能改变实际观察到的入口和 launch 次数, 不能按 GGML node 数推算 kernel 次数。本次未验证 NVIDIA 环境、未观察 kernel launch, 只完成 launcher 边界阅读。

## 7. 当天验收状态

- 已完成: CPU 调用链、type traits 与 activation 转换解释、工作区和线程分块说明、PP/TG shape 对照、两条有条件且可验证的假设。
- 暂时跳过: 实际 MUL_MAT 的 GDB 记录、`mul-mat-support.txt`、数值比较和 CUDA launch 观察。均因本次跳过手动实验, 没有测试通过或性能提升结论。
- 后续补录: 先运行路线中的小用例, 在局部变量初始化后记录 `src0/src1/dst`、`vec_dot_type/from_float`、`ith/nth/use_ref`、工作区和 backtrace; 如果命中优化入口, 记录真实分支。
