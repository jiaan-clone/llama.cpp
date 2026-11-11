# week7 day1: Tensor、type traits、op 和 cgraph

本文回答[第 7 周学习路线](../../007-llama-cpp-week7-daily-learning-route.md)中 Day 1 的当天输出与验收。

核对基准为当前 checkout `697b4bbf4633a38abfd66f7cc54a167095aedef3`, 与路线中的 `545ddb30a` 不同。已阅读源码, 并只读解析本地 GGUF 的 metadata 和 tensor 描述; 未构图运行、未推理、未使用 GDB。按要求跳过手动实验, 下文区分文件事实、静态推导和待实测字段。路线要求的 `tensors.csv` 内容集中放在本文, 未另行创建 `results/week7/` 实验文件。

## 1. Q projection 的语义 shape 与 GGML 布局

本地文件为 `/home/king/llama.cpp/deepseek-r1-1.5b-f16.gguf`。文件中的架构为 `qwen2`, 层数为 28, `n_embd=1536`, `n_head=12`, `n_head_kv=2`, 因而每个 Q head 的维度为 `1536/12=128`。第 0 层 `blk.0.attn_q.weight` 为 F16, GGUF shape 为 `[1536,1536]`; `blk.0.attn_q.bias` 为 F32, shape 为 `[1536]`。

令 `N` 表示本次 ubatch 中的 token 数, 固定普通二维、连续、无 LoRA 的独立 Q 投影路径:

```text
语义: X[N,K] * W[M,K]^T = Y[N,M]
GGML:
  W.ne = [K,M,1,1]
  X.ne = [K,N,1,1]
  Y.ne = [M,N,1,1]
  Y[m,n] = sum_k W[k,m] * X[k,n]

本模型: K=1536, M=n_head*head_dim=1536
  attn_norm -> MUL_MAT(Wq,X) -> ADD(q_bias) -> RESHAPE -> ROPE
```

`ne[0]` 是最内层维度, 上面的数学 shape 采用 token 在前的习惯, 两者不能直接按书写顺序混用。`MUL_MAT` 的输出是 F32; 权重为 F16 不代表 activation 和结果也必须为 F16。

| 对象 | type | `ne[4]` | 连续布局 `nb[4]`, 单位 byte | 逻辑数据大小 |
| --- | --- | --- | --- | ---: |
| Wq | F16 | `[1536,1536,1,1]` | `[2,3072,4718592,4718592]` | 4,718,592 |
| X, norm 后的 hidden state | F32 | `[1536,N,1,1]` | `[4,6144,6144*N,6144*N]` | `6144*N` |
| Y, bias 前的 Q 投影结果 | F32 | `[1536,N,1,1]` | `[4,6144,6144*N,6144*N]` | `6144*N` |
| Q, bias 后拆分 head | F32 | `[128,12,N,1]` | `[4,512,6144,6144*N]` | `6144*N`, 与源数据共享 |

表中的 Wq type/shape 来自文件; stride 与中间 tensor 来自源码推导, 不是实际内存快照。backend 可以使用专用布局或 padding, 不能直接拿此表代替加载后的 `nb/buffer/data` 检查。

### 一个实际源码中的 view 来源

[Qwen2 构图](../../../../src/models/qwen2.cpp)调用 `build_qkv()`; [llama-graph.cpp](../../../../src/llama-graph.cpp) 的独立 Q/K/V 分支先执行 `build_lora_mm(layer.wq, cur, ...)`, 有 bias 时执行 `ggml_add`, 再调用 `ggml_reshape_3d(ctx0, Qcur, 128, 12, N)`。

这个 reshape 是共享数据的 view, `op=GGML_OP_RESHAPE`, `src[0]` 指向 bias 后的 Q, `view_offs=0`。普通非 view 源情况下, `view_src` 也指向这个 Q; 如果来源本身是 view, `ggml_new_tensor_impl()` 会折叠到基底并累加偏移。它没有复制一份 Q 数据。随后 `ggml_rope_ext()` 才建立另一项计算。

源码多次使用 `Qcur-0` 作为回调名称, 所以定位时还要看 `op/src/view_src`, 不能仅凭名字判断命中了 bias 前、bias 后、reshape 还是 RoPE 后的 Q。若模型加载选择了合并 QKV 路径, view 改为从合并结果 `qkv` 中切片; 此处只推导独立 projection 分支, 实际选择待断点核对。

## 2. 小型 MUL_MAT 手算

取 `K=4,M=3,N=2`, 自选便于手算的输入, 每行表示固定 m 或 n 下沿 k 的连续数值:

```text
W[:,0] = [ 1, 2, 3, 4]
W[:,1] = [ 0, 1, 0, 1]
W[:,2] = [-1, 0, 1, 0]
X[:,0] = [1, 0, 2, 1]
X[:,1] = [0, 1, 1, 2]

Y[0,0] = 1*1 + 2*0 + 3*2 + 4*1 = 11
Y[:,0] = [11,1,1]
Y[:,1] = [13,3,1]
Y.ne   = [3,2,1,1]
```

F16 W 的 `nb=[2,8,24,24]`, 大小为 24 bytes; F32 X 的 `nb=[4,16,32,32]`, 大小为 32 bytes; F32 Y 的 `nb=[4,12,24,24]`, 大小为 24 bytes。

将 `N` 改成 1, 只保留一个 token 的输入向量, 输出为 `[3,1,1,1]`, 这对应单 sequence、一次一个新 token 的 TG projection。矩阵 W 没有变小。多个 sequence 同时 decode 时 N 可以大于 1, 不能把所有 decode 都等同于 N=1。

## 3. Tensor 记录表草案

以下可用作 `results/week7/tensors.csv` 的初稿。`static_N1` 表示假设 `N=1` 的静态示例, 不表示真的执行了 decode。`Q_mm_alias` 是本文给 bias 前输出取的别名; `pending` 表示必须在实验后补齐。这里的“输出”指 Q projection 的输出, 不是最终 logits。

```csv
phase,tensor_name,op,src_names,type,ne[4],nb[4],view_src,logical_bytes,buffer_type,assigned_backend,lifetime,evidence
static_N1,blk.0.attn_q.weight,NONE,none,F16,"[1536,1536,1,1]","[2,3072,4718592,4718592]",none_expected,4718592,pending,pending,model,GGUF_type_shape_and_contiguous_layout_derivation
static_N1,attn_norm-0,MUL,"RMS_NORM_result;blk.0.attn_norm.weight",F32,"[1536,1,1,1]","[4,6144,6144,6144]",none_expected,6144,pending,pending,graph_consumers,source_derivation_runtime_pending
static_N1,Q_mm_alias,MUL_MAT,"blk.0.attn_q.weight;attn_norm-0",F32,"[1536,1,1,1]","[4,6144,6144,6144]",none_expected,6144,pending,pending,graph_consumers,source_derivation_runtime_pending
static_N1,Q_reshape_alias,RESHAPE,Q_after_bias_alias,F32,"[128,12,1,1]","[4,512,6144,6144]",Q_after_bias_alias,6144,pending,pending,source_and_view_consumers,source_derivation_runtime_pending
```

中间 tensor 的数据存活到最后一个消费者使用结束, 其 metadata 则可能随 graph reuse 保留得更久。view 不独占表中 6144 bytes, 统计总内存时不能重复累加。

## 4. 构图为何不是执行

[ggml.c](../../../../ggml/src/ggml.c) 的 `ggml_mul_mat()` 检查 shape 兼容性, 创建 F32 结果 tensor, 设置 `op=GGML_OP_MUL_MAT` 和 `src[0/1]`, 然后返回。这里没有 dot product 或 backend compute 调用。即使 `no_alloc=false` 时分配了结果存储, 也不代表算出了结果。

`ggml_build_forward_expand()` 沿 `src` 递归访问依赖, 利用 visited 集合去重, 将依赖排列进 `ggml_cgraph`。普通 `op=NONE` 且非 PARAM 的 tensor 被放入 `leafs`, 带运算的 tensor 被放入 `nodes`; reshape/view 也可能是 node, 但它们不一定对应实际计算 kernel。图中 leaf 的分类也不代表应用永远不能更新其输入数据。

```text
ggml_context arena
  -> tensor metadata: type / ne / nb / op / src / view information
  -> graph metadata: nodes / leafs / use_counts / visited set

tensor.buffer -> backend buffer -> allocated or wrapped storage
tensor.data   -> tensor's location inside storage
view.data     -> base tensor data + view_offs
```

`no_alloc=true` 只禁止 context 为普通 tensor 自动分配数据区, 不禁止分配 tensor/graph metadata。后续可以由 backend allocator 绑定数据; view 的源已经有数据时也可能直接取得有效数据指针。旧式 context 自己分配数据的 tensor 还可能没有 backend buffer, 所以上图描述的是 backend 管理数据的路径。

## 5. 三种大小不能混用

| 大小 | 含义 | 为何不同 |
| --- | --- | --- |
| `sizeof(ggml_tensor)` | C 结构体 metadata 的大小 | 主要由 ABI、字段和编译配置决定, 不随 tensor 元素数成比例增长 |
| tensor 数据大小 | 连续情况下为各行数据字节总和 | 由 type、block size、shape 决定; 不包含对象本身 |
| buffer allocation | backend 实际分配或包装的存储容量 | 可能容纳多个 tensor, 含 alignment、padding、复用空间或预留容量 |

`ggml_nbytes()` 对非连续 tensor 使用 stride 计算地址跨度, 可能包含间隙, 不总等于“元素数乘元素大小”。backend 的 `get_alloc_size()` 还可能要求更大的存储。

对连续 block quantized W, 只有 K 满足 block 对齐时才能使用 `row_bytes=(K/block_size)*type_size`, `tensor_bytes=row_bytes*M`。`type_size` 是整个 block 的字节数; 例如 Q4_0 的 `nb[0]=18` 是 block 存储大小, 不能拿它作为每个逻辑量化值的标量步长。Day 5 给出完整手算。

## 6. 当天验收状态

- 已完成: Q projection 的语义与 `ne/nb` 对照、小矩阵手算、源码中的 reshape view 来源、tensor CSV 草案、构图与执行及三种大小的解释。
- 暂时跳过: 实际 graph、buffer 放置、view 指针和输出数值核对。原因是本次要求跳过手动实验, 不是已经验证通过。
- 后续补录: 在 Day 3 的真实 prefill/decode 断点中确认每个节点的名字、`op/src/ne/nb/view_src/buffer/data` 和 assigned backend, 再将 CSV 中的 `pending` 替换为记录。
