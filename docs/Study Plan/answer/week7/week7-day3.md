# week7 day3: Scheduler、分图、内存规划与复制

本文回答[第 7 周学习路线](../../007-llama-cpp-week7-daily-learning-route.md)中 Day 3 的当天输出与验收。

源码基准为 `697b4bbf4633a38abfd66f7cc54a167095aedef3`。按要求跳过 GDB、模型 decode 和 CPU/GPU 传输实验。因此本文提供源码调用链、条件化 split 图和 trace 模板, 不将它们称为真实断点记录, 也未生成 `results/week7/scheduler-trace.txt`。

## 1. 从 ubatch 到执行的调用链

主路径见 [llama-context.cpp](../../../../src/llama-context.cpp) 与 [ggml-backend.cpp](../../../../ggml/src/ggml-backend.cpp):

```text
llama_decode()
  -> llama_context::decode()
  -> 按内存与 batch 条件处理 ubatch
  -> llama_context::process_ubatch()
       -> mctx->apply()
       -> graph_params() / res->can_reuse()
       -> 不可复用时:
            res->reset() / ggml_backend_sched_reset()
            model.build_graph()
            ggml_backend_sched_alloc_graph()
              -> ggml_backend_sched_split_graph()
              -> ggml_backend_sched_alloc_splits()
       -> res->set_inputs()
       -> llama_context::graph_compute()
            -> 设置线程数及 threadpool
            -> ggml_backend_sched_graph_compute_async()
                 -> 必要时 alloc_graph()
                 -> ggml_backend_sched_compute_splits()
                      -> 准备各 split 的输入与同步依赖
                      -> ggml_backend_graph_compute_async(backend, split_graph)
```

`process_ubatch()` 用 `ubatch.n_tokens > 1` 选择 batch 线程配置。输入数据在建图/分配之后设置; allocator 不负责计算这些值。decode 中的 KV 更新也有跨 token 的状态含义, 不能把所有 tensor 都看作可随时丢弃的临时量。

## 2. Assignment、split、allocation、execution 各做什么

| 阶段 | 输入与决策 | 结果 |
| --- | --- | --- |
| Assignment | 已分配 buffer、用户指定 backend、权重位置、`supports_op`、`supports_buft`、优先级和 offload 偏好 | 每个节点的 backend id |
| Split | 沿图顺序检查执行 backend 和输入兼容性 | 若干 `[i_start,i_end)` 图段, 以及需要复制的输入 |
| Allocation | tensor 大小、alignment、存活区间、view 和跨 split 依赖 | 计算 buffer 的容量及 tensor 对应地址 |
| Execution | 输入数据、copy/event 依赖、各 split 图 | backend 执行并产生结果 |

`ggml_backend_sched_split_graph()` 当前源码有五个主要 pass:

1. 保留已有用户分配, 根据预分配 tensor/buffer 和权重位置分配节点。graph input 默认倾向最后的 CPU backend。预分配数据不能被当成可任意搬迁的空 tensor。
2. 在支持 op 的前提下沿图向前、向后扩展已有放置, 先扩展非 CPU 的分配, 再填其他位置。
3. 给未分配节点选择支持更多输入的 backend; 对已分配节点, 在计算 buffer type 相同且源 buffer 都兼容等条件下尝试更高优先级 backend。
4. 补齐来源节点和 view 的分配。view 与其源共享数据, 不能独立决定一份无关存储。
5. 划分图段并插入需要的输入副本, 将消费者的 `src` 指向对应副本。

这是一组启发式规则, 不是全局最优性能求解器。split 不只是遇到 backend 变化才产生: 同一 backend 上, 不兼容权重的分段复用及输入容量条件也可能触发新 split。

`split->n_inputs` 主要是 scheduler 为该 split 记录的跨 buffer 副本输入数, 不等于模型的 token 输入数, 也不等于所有 op 源操作数的总数。图段可以有很多 op, 一个 op 也可能被 backend 融合或拆成多个 kernel。

## 3. CPU split 图: 源码推演, 真实图暂缺

假设只有一个普通 CPU backend, 权重/输入都在其可访问的 host buffer, 没有需要跨设备复制的数据, 则一个普通非空推理图预期形成以下结构:

```text
CPU weights + CPU inputs
  |
  v
split 0, backend=CPU, nodes=[0,G)
  embedding -> norm -> Q/K/V -> attention -> FFN -> ... -> logits
  |
  v
CPU results

条件化预期: n_splits=1, split copy inputs=0
实际 G、n_splits、backend id、地址、调用栈: 未测
```

这不代表观察到一次 CPU decode。即使 `n_inputs=0`, 也不表示整个程序没有 memcpy: 设置输入、输出整理或 CPU op 内部仍可能复制数据; 它只说明该条件下没有 scheduler 所记录的跨 backend 输入副本。

真实 CPU split 图与 route 要求的断点调用栈本次暂时跳过, 原因是用户要求跳过手动实验。不能用上面的 `1/0` 预期值填进实测列。

## 4. 可选跨设备图: 同样仅为源码推演

```text
split A on CPU
  produces T in host buffer
       |
       +-- GPU supports source buft?
              |
              +-- yes: use T directly, preserve readiness ordering
              |
              +-- no: allocate T_copy in GPU-compatible buffer
                         -> wait until destination copy slot is reusable
                         -> copy T to T_copy with producer dependency
                         -> split B consumes T_copy on GPU
                         -> record completion event when applicable
```

复制条件的核心是源 backend 与目标不同, 且 `ggml_backend_sched_buffer_supported()` 为 false。不同 backend 可以共享兼容 buffer, 所以 backend 边界不必然是物理传输边界。

举例: 如果某个边界 tensor 为连续 F32 `[1536,N,1,1]`, 普通完整 tensor copy 的有效字节数是 `6144*N`; 假设 N=1 时为 6144 bytes。这只是体积推导, 不是本机 PCIe 流量、传输次数或耗时。真实案例还要记录 tensor 名称、view、源/目标 buft、copy 入口和同步原因。

当前 `ggml_backend_sched_compute_splits()` 对复制有不同处理:

| 情况 | 源码行为 | 目的 |
| --- | --- | --- |
| 被标为 `GGML_TENSOR_FLAG_INPUT` 且需要复制的用户输入 | 等待目标 event 或同步目标 backend, 然后立即 `ggml_backend_tensor_copy()` | 避免用户在延迟复制之前改写源数据 |
| 普通 split 输入 | 等待目标复用槽可用, 尝试 backend 的 `cpy_tensor_async` | 在 backend 能正确处理依赖时安排异步传输 |
| 异步 copy 不支持 | 同步源, 再等待目标 event 或同步目标, 使用普通 tensor copy | 保证源已产生、目标不再被前次计算使用 |
| split 计算完成提交后 | 在有输入及 event 的条件下记录 event | 约束下一次覆盖该 copy 槽的时机 |

有 eval callback 时, scheduler 还可能把 split 按观察点进一步分段执行并同步后回调。MoE 权重另有只复制所用 experts 的路径; 本文的 Qwen2 dense 示例不据此假定存在 expert 传输。

## 5. Compute buffer 为什么可以复用

[ggml-alloc.c](../../../../ggml/src/ggml-alloc.c) 的 gallocr 跟踪源 tensor 的消费者数 `n_children`、存活 view 数 `n_views`, 以及 OUTPUT 等保留条件。它在规划中模拟依赖消耗, 释放不再被使用的临时存储位置, 并在满足布局与引用条件时允许 inplace 复用。

```text
示意, 不是本模型实测地址:
T0 --consumed by--> T1 --consumed by--> T2

compute region A: [T0 live] ........ [T2 live]
compute region B: ....... [T1 live] ........
```

若 T0 已无后续消费者和活 view, T2 可以使用同一段空间; 如果另一个残差分支或 view 还要读取 T0, 则不能提前覆盖。这里的“释放”常是图规划中归还 offset, 不代表每个 op 后都调用系统 free。compute buffer 容量接近带对齐和保留条件的峰值需求, 不是所有中间 tensor 字节数总和。

`ggml_gallocr_reserve_n()` 用图和 backend id 规划并准备所需 buffer; `ggml_gallocr_alloc_graph()` 根据保存的分配信息初始化 tensor 地址。容量或分配条件变化时, scheduler 可能先同步所有 backend, 再 reserve/alloc, 防止仍在执行的任务引用被移动的地址。

权重通常活到 model 销毁, KV 数据要供后续 token 使用, 最终输出要满足读取契约; 它们不能套用“一个算子执行完就复用”的规则。CPU backend 还有自己的 kernel work buffer, 与 gallocr 管理的图 tensor buffer 是不同的用途。

## 6. Reserve、prefill、decode 与 graph reuse

| 命中阶段 | 作用 | 如何区分 |
| --- | --- | --- |
| reserve | 为容量估算/预留而构建的测量图 | 看初始化/reserve 调用栈, 不是一次实际 prompt 的输出 |
| prefill | 对实际 prompt token 的 ubatch 计算 | 看 decode/process_ubatch 栈、真实 token 数和输入 |
| decode | 通常消费后续新 token | 记录 sequence、`ubatch.n_tokens`、KV 状态和图参数 |

graph reuse 是复用兼容图结构及已分配状态, 每次仍要设置新输入并运行计算。当前 `process_ubatch()` 在 `can_reuse()` 为真时不重建图; pipeline parallel 开启时还会先同步, 防止 `set_inputs()` 覆盖上一轮 GPU 正在读取的输入。因此不能要求每个 token 都命中 split 和 allocation 断点。

异步提交返回只表示该 API 的提交阶段结束, 不构成所有 backend 已完成的通用保证。直接读取结果前要经过对应同步/结果获取接口; 复用输入前要确认消费者不再读取它。CPU 目前可能在调用中完成计算, 但这个特性不能推广到 GPU。用户输入的立即复制保护也不能推出“所有输入、任何时候都能覆盖”。

## 7. scheduler-trace.txt 待补模板与验收状态

```text
status=SKIPPED_MANUAL_EXPERIMENT
source_commit=697b4bbf4633a38abfd66f7cc54a167095aedef3
model=deepseek-r1-1.5b-f16.gguf
requested_config=CPU_only; ngl=0; Debug
observed_config=pending

phase=reserve
backtrace=pending
graph_nodes=pending; n_backends=pending; n_splits=pending

phase=prefill
backtrace=pending
ubatch_tokens=pending; graph_reused=pending
split_id=pending; backend_id=pending; backend_name=pending
i_start=pending; i_end=pending; n_inputs=pending; graph_nodes=pending

phase=decode
backtrace=pending
ubatch_tokens=pending; graph_reused=pending
split_id=pending; backend_id=pending; backend_name=pending
i_start=pending; i_end=pending; n_inputs=pending; graph_nodes=pending

selected_mul_mat=pending
src_names=pending; types=pending; ne=pending; nb=pending
view_src=pending; buffer_types=pending; data_addresses=pending
copy_src=pending; copy_dst=pending; copy_bytes=pending
copy_entry=pending; synchronization_reason=pending
```

后续按路线先断 `llama_context::graph_compute`, 再断 `ggml_backend_sched_compute_splits`; 需要观察规划时另开运行断 split/alloc。已完成静态路径、内存复用和异步边界解释; 真实 split 图、断点 trace 及传输验证均暂时跳过, 不判为实验验收通过。
