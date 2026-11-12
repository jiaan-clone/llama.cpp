# week3 day1

## 习题1

题目: 公式表、两个手算例子、逻辑 shape 与 `ne[]` 对照。

以下按普通 dense Llama 的单 sequence、单 KV stream、causal attention 解释, 假定 `D = H * Dh`, Q/K/V 的 head dimension 都为 `Dh`, 且 `H` 能被 `KvH` 整除。公式省略 optional bias、LoRA、特殊缩放和 control vector, attention 的布局表采用非 Flash Attention 路径。

源码核对基准为当前工作区 `dd545f7e7`, 与学习路线记录的 `545ddb30a` 不同。下面的数值是纸面计算, `ne[]` 是根据当前源码推导的布局, 不是模型运行日志。

### 1. 符号与 shape 约定

| 符号 | 含义 |
| --- | --- |
| `N` | 当前 ubatch 输入的 token 数, 也是本次 query 数 |
| `S` | 本次 attention 涉及的有效历史及当前 token 总数; 每个 query 能访问哪些位置还取决于 causal mask |
| `Kcap` | 当前 graph 读取的 KV 范围长度, 对应 `n_kv`; 可含 padding 或空槽, 不等于有效长度 `S`, 也不必等于 `n_ctx` |
| `O` | 当前 ubatch 请求输出的 token 位置数, 通常 `O <= N` |
| `D` | hidden size |
| `H / KvH / Dh` | Q head 数 / KV head 数 / 每个 head 的特征维度 |
| `I / V / L` | FFN 中间宽度 / 词表大小 / Transformer block 数 |

纸面 batch size 固定为 1, 下文省略这个轴。激活通常按 `[token, feature]` 写, 线性权重按 `[out, in]` 写, 因此线性层公式为 `Y = X W^T`。注意力分数先按每个 head 的 `[N,S]` 理解, 合并记为 `[H,N,S]`。

`V` 单独作为维度符号时表示词表大小; value 张量写成 `Vcur` 或 `Vcache`, 避免混淆。GGML 的二维 `ne=[a,b]` 是 `ne[0..3]=[a,b,1,1]` 的简写。

### 2. 单层公式表

对输入 `X` 的第 `n` 个 token, RMSNorm 为:

$$
\operatorname{RMSNorm}(X)_{n,d}
=\frac{X_{n,d}}{\sqrt{\frac{1}{D}\sum_{j=0}^{D-1}X_{n,j}^{2}+\epsilon}}\,\gamma_d.
$$

它只沿该 token 的 `D` 个特征求平方均值, 不跨 token, 也不减去均值。attention norm、FFN norm 和最终 output norm 各有自己的权重 `gamma`。

令 `g = H / KvH`, 第 `h` 个 Q head 使用的 KV head 为 `j(h) = floor(h/g)`。下表的 K/V cache 按有效位置描述, 暂不把 padding 算入 `S`。

| 阶段 | 数学公式或操作 | 逻辑输出 shape | 当前源码中的主要入口 |
| --- | --- | --- | --- |
| embedding | `X = E[token_ids]` | `[N,D]` | `build_inp_embd()` / `ggml_get_rows()` |
| attention norm | `Xn = RMSNorm_attn(X)` | `[N,D]` | `build_norm(..., LLM_NORM_RMS, ...)` |
| Q 投影 | `Qflat = Xn Wq^T` | `[N,H*Dh]` | `build_qkv()` / `build_lora_mm()` |
| K 投影 | `Kflat = Xn Wk^T` | `[N,KvH*Dh]` | 同上 |
| V 投影 | `Vflat = Xn Wv^T` | `[N,KvH*Dh]` | 同上 |
| 拆 head | 将投影特征拆成 head 和 head 内特征 | Q `[N,H,Dh]`, K/V `[N,KvH,Dh]` | `ggml_reshape_3d()`; 合并 QKV 路径用 `ggml_view_3d()` |
| RoPE | `Qcur = RoPE(Q,pos)`, `Kcur = RoPE(K,pos)`; `Vcur` 不旋转 | shape 不变 | `ggml_rope_ext()` |
| KV cache | 写入本次 `Kcur/Vcur`, 读取历史及当前 K/V | 有效内容均为 `[S,KvH,Dh]` | 带 KV cache 的 `build_attn()` |
| attention score | `A_h = Qcur_h Kcache_j(h)^T / sqrt(Dh) + M` | 每个 head `[N,S]`, 合起来 `[H,N,S]` | `build_attn_mha()` / `ggml_mul_mat(k,q)` + `ggml_soft_max_ext()` 的 scale、mask 部分 |
| softmax | `P_h[n,s] = exp(A_h[n,s]) / sum_r exp(A_h[n,r])` | `[H,N,S]` | `ggml_soft_max_ext()` |
| 加权 V | `Z_h = P_h Vcache_j(h)` | 每个 head `[N,Dh]` | `ggml_mul_mat(v,kq)`; 此时 `kq` 已是概率 |
| 拼接与输出投影 | `Z = concat_heads(Z_h)`, `Attn = Z Wo^T` | Z `[N,H*Dh]`, Attn `[N,D]` | `permute + cont_2d` 后接 `build_lora_mm()` |
| 第一次 residual | `R = X + Attn` | `[N,D]` | `ggml_add()` |
| FFN norm | `Rn = RMSNorm_ffn(R)` | `[N,D]` | `build_norm()` |
| gate / up 投影 | `G = SiLU(Rn Wgate^T)`, `U = Rn Wup^T` | 均为 `[N,I]` | `build_ffn()` |
| SwiGLU 与 down 投影 | `F = (G .* U) Wdown^T` | 门控乘积 `[N,I]`, F `[N,D]` | 普通路径 `ggml_swiglu_split()` 后接 down 投影 |
| 第二次 residual | `Y = R + F` | `[N,D]` | `ggml_add()` |
| 最终 output norm | 最后一个 block 后, `Ynorm = RMSNorm_out(Y_selected)` | `[O,D]` | `build_norm()` |
| lm head | `logits = Ynorm Wout^T` | `[O,V]` | `build_lora_mm(model.output, ...)` |

其中 `SiLU(z) = z / (1 + exp(-z))`, `.*` 表示逐元素乘法。`Wo` 是每层 attention 的输出投影, `Wout` 是所有 block 之后的词表投影, 两者不同。embedding 和最终 output norm / lm head 也不属于每个 block。

RoPE 根据 position 旋转 Q/K 的特征对, 保留 shape, 使点积携带位置信息; causal mask 负责禁止读取未来位置。二者作用不同, 使用 RoPE 后仍需要 causal mask。

**最后一层选行:** 上表先按所有 token 写完整公式。当前 Llama 源码会在最后一层 attention 输出投影后, 对 `Attn` 和 residual 输入 `X` 同时用 `inp_out_ids` 选行, 再执行第一次 residual。因此该层从 `R` 开始, norm、FFN、第二次 residual 的 token 维度都可由 `N` 变为 `O`。这些操作逐 token 独立, 提前选行与先算完整个 block 再选出对应行等价; attention 仍需处理其他 token 提供的 K/V。

**权重、激活和状态:** `E`、各投影矩阵及 norm 的 `gamma` 是推理时固定的模型权重; `X/Q/P/G/U/logits` 等是激活; KV cache 是跨推理调用保留的状态, 普通路径存入每层经过 RoPE 的 K 和未经过 RoPE 的 V, 不存 Q。

### 3. 手算例子一: 二维线性投影

给定逻辑矩阵:

```text
X = [[1, 2, 3],
     [4, 5, 6]]       shape = [N,Din] = [2,3]

W = [[1, 0,  1],
     [0, 1, -1]]      shape = [Dout,Din] = [2,3]

W^T = [[1,  0],
       [0,  1],
       [1, -1]]       shape = [Din,Dout] = [3,2]
```

逐项计算 `Y = X W^T`:

```text
Y[0,0] = 1*1 + 2*0 + 3*1    =  4
Y[0,1] = 1*0 + 2*1 + 3*(-1) = -1
Y[1,0] = 4*1 + 5*0 + 6*1    = 10
Y[1,1] = 4*0 + 5*1 + 6*(-1) = -1

Y = [[ 4, -1],
     [10, -1]]        shape = [N,Dout] = [2,2]
```

对应 GGML:

| 张量 | 逻辑 shape | `ne[0..3]` | `ne[0]` 的语义 |
| --- | --- | --- | --- |
| X | `[N,Din]=[2,3]` | `[3,2,1,1]` | 输入特征 |
| W | `[Dout,Din]=[2,3]` | `[3,2,1,1]` | 每个输出通道权重的输入特征 |
| `ggml_mul_mat(ctx,W,X)` | `[N,Dout]=[2,2]` | `[2,2,1,1]` | 输出特征 |

调用者不需要显式构造 `W^T` tensor, 因为 `ggml_mul_mat()` 的语义已经是让两个输入沿共同的 `ne[0]` 做点积。用 GGML 轴索引表示就是:

```text
Y_ggml[o,n] = sum_i W_ggml[i,o] * X_ggml[i,n]
```

用本文的逻辑索引表示, 它正是 `Y[n,o] = sum_i X[n,i] * W[o,i]`。这来自算子的索引约定, 并不要求额外创建一个转置节点。

### 4. 手算例子二: 两个 token 的 causal attention

设 `H=KvH=1`, `Dh=2`, `N=S=2`, 没有历史 token。纸面计算不引入 padding, 所以取 `Kcap=2`。Q/K 已完成 norm、投影和 RoPE, 本例只计算单 head 的注意力输出, 不继续计算 `Wo` 或 residual。

```text
Q = K = [[1, 0],
         [0, 1]]

Vcur = [[1, 2],
        [3, 4]]

M = [[0, -inf],
     [0,    0]]
```

**第一步: 点积与缩放。** 每一行对应一个 query, 每一列对应一个 key。

```text
Q K^T = [[1*1 + 0*0, 1*0 + 0*1],
         [0*1 + 1*0, 0*0 + 1*1]]
      = [[1, 0],
         [0, 1]]

1/sqrt(Dh) = 1/sqrt(2) = 0.7071067812

Q K^T / sqrt(2) = [[0.7071067812, 0],
                  [0, 0.7071067812]]
```

**第二步: 加 causal mask。** 第 0 个 query 不能访问未来的第 1 个 key。

```text
A = Q K^T / sqrt(2) + M
  = [[0.7071067812,         -inf],
     [0,           0.7071067812]]
```

**第三步: 对每一行的 key 位置做 softmax。** 令 `a = 1/sqrt(2)`, 则 `exp(a) = 2.0281149816`。

```text
P[0] = [exp(a)/(exp(a)+0), 0/(exp(a)+0)]
     = [1, 0]

P[1] = [1/(1+exp(a)), exp(a)/(1+exp(a))]
     = [0.3302384507, 0.6697615493]
```

这里 `exp(-inf)=0`。每行的概率之和为 1, 表示固定 query 对所有允许访问的 key 分配注意力权重。实际实现可先减去该行最大值以提高数值稳定性, 不改变数学结果。

**第四步: 用概率加权 V。**

```text
Z[0] = 1*[1,2] + 0*[3,4]
     = [1,2]

Z[1,0] = 0.3302384507*1 + 0.6697615493*3 = 2.3395230987
Z[1,1] = 0.3302384507*2 + 0.6697615493*4 = 3.3395230987

Z = [[1,      2     ],
     [2.3395, 3.3395]]
```

softmax 沿 key 位置归一化, 是因为输出是“一个 query 对各个 value 的加权和”。若沿 query 归一化, 就无法保证每个 query 的权重之和为 1。第 0 个 token 不能使用第 1 个 V, 是因为对应 key 的分数被 mask 成 `-inf`, 概率严格为 0; 不是因为该 V 没有算出来。

本例对应的 GGML 布局如下。虽然多个维度的长度恰好都是 2, 仍要区分每个轴的含义。

| 阶段 | 逻辑 shape | `ne[0..3]` 及轴含义 |
| --- | --- | --- |
| Q 刚拆 head | `[N,H,Dh]=[2,1,2]` | `[2,1,2,1] = [Dh,H,N,1]` |
| matmul 使用的 q | 每个 head `[N,Dh]=[2,2]` | `[2,2,1,1] = [Dh,N,H,1]` |
| matmul 使用的 k | 每个 KV head `[S,Dh]=[2,2]` | `[2,2,1,1] = [Dh,Kcap,KvH,1]` |
| 原始点积 `kq` | `[H,N,S]=[1,2,2]` | `[2,2,1,1] = [Kcap,N,H,1]` |
| mask | `[N,S]=[2,2]` | `[2,2,1,1] = [Kcap,N,1,1]` |
| scale + mask + softmax 后的 P | `[H,N,S]=[1,2,2]` | `[2,2,1,1]`, 沿 `ne[0]` 归一化 |
| 第二次 matmul 使用的 v | value 语义仍为 `[S,KvH,Dh]` | `[2,2,1,1] = [Kcap,Dh,KvH,1]` |
| `ggml_mul_mat(v,P)` | 每个 head `[N,Dh]=[2,2]` | `[2,2,1,1] = [Dh,N,H,1]` |

尤其注意第二次 matmul 的 v: 对固定特征 `d=0`, 沿 `ne[0]` 取到的是不同 key 位置的 `[1,3]`; 对 `d=1` 取到 `[2,4]`。因此其运算用的二维行视图相当于 `Vcur^T = [[1,3],[2,4]]`, 才能沿 key 位置与 P 做点积。shape 数字相同不代表布局相同。

### 5. 一般情况下的逻辑 shape 与 `ne[]` 对照

#### 5.1 权重与普通二维激活

| 张量 | 逻辑 shape | GGML `ne[]`, 省略尾部的 1 |
| --- | --- | --- |
| embedding 表 E | `[V,D]` | `[D,V]` |
| attention / FFN / output norm 的 gamma | `[D]` | `[D]`, 沿 token 轴广播 |
| Wq | `[H*Dh,D]` | `[D,H*Dh]` |
| Wk、Wv | `[KvH*Dh,D]` | `[D,KvH*Dh]` |
| Wo | `[D,H*Dh]` | `[H*Dh,D]` |
| Wgate、Wup | `[I,D]` | `[D,I]` |
| Wdown | `[D,I]` | `[I,D]` |
| Wout | `[V,D]` | `[D,V]` |
| X、Xn、R、Rn、F、Y | `[N,D]` | `[D,N]` |
| gate / up / 门控乘积 | `[N,I]` | `[I,N]` |
| 最后一层选行后的 hidden | `[O,D]` | `[D,O]` |
| logits | `[O,V]` | `[V,O]` |

最后一层提前选行后, 表中对应的 FFN 激活应将 `N` 替换为 `O`。Wout 可以与 embedding 权重共享, 但两者承担的计算角色仍然不同。

#### 5.2 attention 的轴变化

下表中逻辑 K/V 和 score 用有效长度 `S` 描述; GGML 使用 `Kcap`, 多出来的无效位置由 mask 排除。Q/K/V 刚拆 head 时只包含本次输入的 `N` 个 token, 进入 attention 的 k/v 则来自含历史的 cache。

| 位置 | 逻辑语义 / shape | GGML `ne[0..3]` |
| --- | --- | --- |
| Q 投影后, 尚未拆 head | `[N,H*Dh]` | `[H*Dh,N,1,1]` |
| K/V 投影后, 尚未拆 head | `[N,KvH*Dh]` | `[KvH*Dh,N,1,1]` |
| Q 拆 head, RoPE 前后 | `[N,H,Dh]` | `[Dh,H,N,1]` |
| K/V 拆 head; K 再做 RoPE | `[N,KvH,Dh]` | `[Dh,KvH,N,1]` |
| attention 的 q | 同一组 Q, 为每个 head 组织 query | `[Dh,N,H,1]` |
| attention 的 k | cache 有效内容 `[S,KvH,Dh]` | `[Dh,Kcap,KvH,1]` |
| `kq = ggml_mul_mat(k,q)` | 未缩放分数 `[H,N,S]` | `[Kcap,N,H,1]` |
| `kq_mask` | 对 head 共享的 mask `[N,S]` | `[Kcap,N,1,1]` |
| `ggml_soft_max_ext(kq,mask,scale,...)` | P `[H,N,S]` | `[Kcap,N,H,1]` |
| 与 P 相乘时的 v | cache 有效内容 `[S,KvH,Dh]`, 调整了轴顺序 | `[Kcap,Dh,KvH,1]` |
| `kqv = ggml_mul_mat(v,P)` | 每个 head 输出 `[N,Dh]` | `[Dh,N,H,1]` |
| kqv 换轴后 | 按 token 拼接 head 的准备布局 | `[Dh,H,N,1]` |
| `ggml_cont_2d()` 拼接 heads | `[N,H*Dh]` | `[H*Dh,N,1,1]` |
| Wo 投影后 | `[N,D]` | `[D,N,1,1]` |

因此三种常见 matmul 可用同一个规则理解: 两个输入的 `ne[0]` 相等并被收缩, 输出的前两维是 `[a.ne[1],b.ne[1]]`。

```text
普通投影: ggml_mul_mat(W, X)
          [Din,Dout] 和 [Din,N] -> [Dout,N], 收缩 Din

QK 点积: ggml_mul_mat(k, q)
          [Dh,Kcap,KvH,1] 和 [Dh,N,H,1] -> [Kcap,N,H,1], 收缩 Dh

加权 V:  ggml_mul_mat(v, P)
          [Kcap,Dh,KvH,1] 和 [Kcap,N,H,1] -> [Dh,N,H,1], 收缩 Kcap
```

后两种运算还按组共享 KV head。`ggml_mul_mat()` 要求第二个输入的 `ne[2]`、`ne[3]` 能分别被第一个输入的对应维度整除; 本例分别对应 `H % KvH == 0` 和单 stream。名称 `kq` 来自 GGML 的传参顺序, 其数学含义仍是 `Q K^T`。

#### 5.3 为什么不能把所有逻辑 shape 机械倒序

刚拆 head 的 Q, 逻辑 shape `[N,H,Dh]` 对应 `ne=[Dh,H,N,1]`。为 attention 换轴后, Q 的 token/head/feature 语义没有改变, `ne` 却成为 `[Dh,N,H,1]`。必须结合计算阶段确认每个轴的意义。

`ne[]` 只说明各轴长度, `nb[]` 才说明沿该轴前进一步的字节跨度。例如假定 Q 是无 padding 的连续 F32 tensor:

```text
拆 head 后:
ne = [Dh, H, N, 1]
nb = [4, 4*Dh, 4*Dh*H, 4*Dh*H*N]

ggml_permute(ctx,Q,0,2,1,3) 后:
ne = [Dh, N, H, 1]
nb = [4, 4*Dh*H, 4*Dh, 4*Dh*H*N]
```

`permute` 同时重排 `ne/nb`, 使用原数据的 view, 本身不复制元素。访问普通 F32 元素时, 相对该 tensor 的 `data` 的字节偏移是 `i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]`。需要连续输出时使用 `cont` 等操作在执行时物化; 量化类型按 block 存储, 不能直接套用这里的 4 字节逐元素公式。

### 6. Day1 验收问题

**RMSNorm 与 softmax 分别沿哪个轴?** RMSNorm 对每个 token 的 hidden 特征求平方均值, 在逻辑 `[N,D]` 中沿 D, 在 GGML `[D,N]` 中沿 `ne[0]`。attention softmax 对固定 head、固定 query 的所有 key 分数归一化, 在逻辑 `[H,N,S]` 中沿 S, 在 GGML `[Kcap,N,H,1]` 中也沿 `ne[0]`。两者都可能操作 `ne[0]`, 但该轴代表的对象完全不同。

**Q head 与 KV head 有什么区别?** Q head 产生各自的 query 和注意力分布; 每个 KV head 提供可被一组 Q heads 共用的 key/value。设 `H=8,KvH=2`, 则 `g=4`, Q heads `0,1,2,3` 使用 KV head 0, Q heads `4,5,6,7` 使用 KV head 1。共享 K/V 不意味着共享 Q 或概率 P。`H=KvH` 是普通 MHA, `KvH=1` 是 MQA, 中间的分组共享是 GQA, 不必先把 KV cache 物理复制成 H 份。

**N、S、O 为什么不是同一个数?** 它们分别计数本次输入、attention 使用的有效历史及当前位置、请求输出的位置。例如已有 4 个历史 token, 本次输入 3 个新 token, 只请求最后一个位置的 logits, 则 `N=3,S=7,O=1`; 三个 query 在普通 causal mask 下分别可访问 5、6、7 个有效位置。首次 prefill 无历史时可以 `N=S`, 但 O 仍可为 1; 逐 token decode 时常为 `N=O=1`, S 随有效历史增长。`Kcap` 还可能因 padding 大于 S。

**如何用一句话解释 helper 的输入输出?**

| 结构 / helper | 输入与输出的含义 |
| --- | --- |
| `llm_graph_params` | 汇集本次构图所需的架构、超参数、ubatch、KV memory context、输出数和 callback 等参数 |
| `llm_graph_context` | 从这些参数建立构图上下文, 持有 GGML context / graph, 提供各类 `build_*` helper |
| `build_norm()` | 输入激活、norm 权重和 norm 类型, 输出同 shape 的归一化激活; RMS 路径包含 `RMS_NORM` 后的权重 `MUL` |
| `build_qkv()` | 输入归一化 hidden 和层权重, 输出已经投影并拆 head 的 Q/K/V; Llama 在其外部给 Q/K 加 RoPE |
| `build_attn()` 的 KV 重载 | 输入 Q/K/V、KV/mask 上下文和 Wo 等参数, 构建 cache 写入与读取、attention 和输出投影, 返回 `[D,N]` 激活 |
| `build_ffn()` | 输入 FFN norm 后的 hidden 和 gate/up/down 权重, dense SwiGLU 路径输出同 token 数的 `[D,N]` 激活; residual 在外部相加 |
| `ggml_tensor` | 用 `ne/nb` 描述 shape 与布局, 用 `op/src[]` 描述计算及依赖, 并关联实际数据存储 |

这些 `build_*` 调用是在构造 tensor 和计算依赖, 不代表此时已完成数值计算。纸面的一个阶段也未必对应一个 op: scale、mask、softmax 可合成 `SOFT_MAX`, `SiLU(gate) .* up` 可由 `ggml_swiglu_split()` 表达。

### 7. 源码核对入口

- [Llama 模型及 graph](../../../../src/models/llama.cpp): `load_arch_tensors()` 核对权重 shape; `graph<embed>::graph()` 核对两次 residual、RoPE、最后一层选行和最终 logits。
- [graph 声明](../../../../src/llama-graph.h): `llm_graph_params`、`llm_graph_context` 及各 helper 的输入输出声明。
- [graph helper 实现](../../../../src/llama-graph.cpp): `build_norm()`、`build_qkv()`、`build_attn_mha()`、带 KV cache 的 `build_attn()` 和 `build_ffn()`。
- [GGML tensor 与 API](../../../../ggml/include/ggml.h): `ggml_tensor` 的 `ne/nb/op/src` 和 `ggml_mul_mat()` 的矩阵约定。
- [GGML 构图实现](../../../../ggml/src/ggml.c): `ggml_can_mul_mat()`、`ggml_mul_mat()`、`ggml_reshape_3d()`、`ggml_permute()` 的维度、stride 和依赖处理。
- [CPU matmul 实现](../../../../ggml/src/ggml-cpu/ggml-cpu.c): 用 `r2 = ne12 / ne02`、`i02 = i12 / r2` 核对 Q head 到 KV head 的分组映射。

## 习题2， 做二维投影小计算
已知：

\\[ X=\begin{bmatrix} 1&2&3\\\ 4&5&6 \end{bmatrix} \\]

\\[ W=\begin{bmatrix} 1&0&1\\\ 0&1&-1 \end{bmatrix} \\]

要求计算 \\(XW^T\\)。

### 第一步：计算 W 的转置

将 \\(W\\) 的行和列互换：

\\[ W^T=\begin{bmatrix} 1&0\\\ 0&1\\\ 1&-1 \end{bmatrix} \\]

### 第二步：矩阵乘法

\\[ XW^T= \begin{bmatrix} 1&2&3\\\ 4&5&6 \end{bmatrix} \begin{bmatrix} 1&0\\\ 0&1\\\ 1&-1 \end{bmatrix} \\]

矩阵乘法规则：左矩阵的每一行，与右矩阵的每一列对应元素相乘再求和。

逐个计算：

\\[ \begin{aligned} y\_{11}&=1\times1+2\times0+3\times1=4\\\ y\_{12}&=1\times0+2\times1+3\times(-1)=-1\\\ y\_{21}&=4\times1+5\times0+6\times1=10\\\ y\_{22}&=4\times0+5\times1+6\times(-1)=-1 \end{aligned} \\]

### 第三步：最终结果

\\[ \boxed{ XW^T= \begin{bmatrix} 4&-1\\\ 10&-1 \end{bmatrix} } \\]

### 第四步：理解矩阵维度

- \\(X\\) 的形状是 \\(2\times3\\)。
- \\(W\\) 的形状是 \\(2\times3\\)。
- \\(W^T\\) 的形状是 \\(3\times2\\)。
- \\(XW^T\\) 的形状是 \\(2\times2\\)。

因此：

\\[ \underbrace{(2\times3)}\_X \cdot \underbrace{(3\times2)}\_{W^T} = \underbrace{(2\times2)}\_{\text{结果}} \\]

补充： 在神经网络中，\\(XW^T\\) 是常见的线性层计算形式，其中 \\(X\\) 表示输入特征，\\(W\\) 表示权重矩阵。每一行输入与每一行权重进行点积，得到对应的输出值。

###  写出 ne[]
在 GGML 中，`ne[]` 表示 Tensor 各维度的元素数量，采用低维在前的存储方式。

对于矩阵：

\\[ X=\begin{bmatrix}1&2&3\\\4&5&6\end{bmatrix},\quad W=\begin{bmatrix}1&0&1\\\0&1&-1\end{bmatrix} \\]

计算：

\\[ Y=XW^T= \begin{bmatrix}4&-1\\\10&-1\end{bmatrix} \\]

对应的 GGML 维度：

| Tensor | 数学矩阵形状 | GGML `ne[]`    |
| ------ | ------ | -------------- |
| X      | 2 × 3  | `{3, 2, 1, 1}` |
| W      | 2 × 3  | `{3, 2, 1, 1}` |
| Y      | 2 × 2  | `{2, 2, 1, 1}` |

其中：

- `ne[0]`：矩阵列数
- `ne[1]`：矩阵行数
- `ne[2]`、`ne[3]`：额外维度，本例均为 1

#### GGML 的 `ggml_mul_mat` 调用

需要注意：GGML 的矩阵乘法参数顺序与数学表达式不同。

```
struct ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
```

该调用对应数学表达式：

\\[ Y=XW^T \\]

其中：

```
// src0 = W
W->ne = {3, 2, 1, 1};

// src1 = X
X->ne = {3, 2, 1, 1};

// dst = Y
Y->ne = {2, 2, 1, 1};
```

GGML 内部输出维度计算规则为：

```
ne[0] = src0->ne[1];
ne[1] = src1->ne[1];
ne[2] = src1->ne[2];
ne[3] = src1->ne[3];
```

在机器学习中，通常约定：
- 行 = 样本数量
- 列 = 特征维度
- 从 3 维投影到 2 维，就是将每个样本的特征数量从 3 个减少到 2 个。
注意：“维”本身不等于“列”，只是这里采用了“每行一个样本”的数据组织方式。

## 习题3 做 causal attention 小计算
暂时跳过 norm、投影和 RoPE, 将下面的 Q/K 视为已经完成这些步骤的单个 head。设 `Dh=2`, 手算 score、mask、softmax 和输出。

   ```text
   Q = K = [[1,0], [0,1]]
   Vcur  = [[1,2], [3,4]]
   M     = [[0,-inf], [0,0]]
   ```
答：
已知：

\\[ Q=K=\begin{bmatrix}1&0\\\0&1\end{bmatrix},\quad V=\begin{bmatrix}1&2\\\3&4\end{bmatrix} \\]

\\[ M=\begin{bmatrix}0&-\infty\\\0&0\end{bmatrix},\quad D_h=2 \\]

Causal Attention 的计算公式：

\\[ \boxed{O=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{D_h}}+M\right)V} \\]

### 1. 计算 Score

\\[ S=\frac{QK^T}{\sqrt{2}} \\]

由于 \\(Q=K\\) 都是单位矩阵：

\\[ QK^T=\begin{bmatrix}1&0\\\0&1\end{bmatrix} \\]

所以：

\\[ \boxed{S=\begin{bmatrix}0.7071&0\\\0&0.7071\end{bmatrix}} \\]

### 2. 添加 Causal Mask

\\[ S\_{\text{masked}}=S+M \\]

\\[ =\begin{bmatrix}0.7071&0\\\0&0.7071\end{bmatrix} +\begin{bmatrix}0&-\infty\\\0&0\end{bmatrix} \\]

\\[ \boxed{S\_{\text{masked}}=\begin{bmatrix}0.7071&-\infty\\\0&0.7071\end{bmatrix}} \\]

这里：

- Token 0 只能关注 Token 0。
- Token 1 可以关注 Token 0 和 Token 1。

### 3. 计算 Softmax

对每一行分别进行 Softmax：

\\[ \operatorname{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}} \\]

第一行：

\\[ \operatorname{softmax}([0.7071,-\infty])=[1,0] \\]

第二行：

\\[ \operatorname{softmax}([0,0.7071]) \\]

因为：

\\[ e^0=1,\qquad e^{0.7071}\approx2.0281 \\]

所以：

\\[ p_0=\frac{1}{1+2.0281}\approx0.3302 \\]

\\[ p_1=\frac{2.0281}{1+2.0281}\approx0.6698 \\]

得到 Attention 权重：

\\[ \boxed{A=\begin{bmatrix}1&0\\\0.3302&0.6698\end{bmatrix}} \\]

### 4. 计算最终输出

\\[ O=AV \\]

\\[ =\begin{bmatrix}1&0\\\0.3302&0.6698\end{bmatrix} \begin{bmatrix}1&2\\\3&4\end{bmatrix} \\]

第一行：

\\[ O_0=[1\times1+0\times3,\quad1\times2+0\times4] \\]

\\[ O_0=[1,2] \\]

第二行：

\\[ O_1=[0.3302\times1+0.6698\times3,\quad 0.3302\times2+0.6698\times4] \\]

\\[ O_1\approx[2.3395,3.3395] \\]

最终：

\\[ \boxed{O\approx\begin{bmatrix}1&2\\\2.3395&3.3395\end{bmatrix}} \\]

### 5. 汇总

| 步骤      | 结果                                                          |
| ------- | ----------------------------------------------------------- |
| Score   | \\(\begin{bmatrix}0.7071&0\\\0&0.7071\end{bmatrix}\\)       |
| Mask 后  | \\(\begin{bmatrix}0.7071&-\infty\\\0&0.7071\end{bmatrix}\\) |
| Softmax | \\(\begin{bmatrix}1&0\\\0.3302&0.6698\end{bmatrix}\\)       |
| Output  | \\(\begin{bmatrix}1&2\\\2.3395&3.3395\end{bmatrix}\\)       |

核心理解：

- 第一个 Token 只能看到自己，因此输出就是 \\(V_0=[1,2]\\)。
- 第二个 Token 能看到前两个 Token，因此输出是 \\(V_0\\) 和 \\(V_1\\) 的加权和。
- Causal Mask 保证每个 Token 不能关注未来的 Token，这是自回归语言模型的重要机制。


解释 softmax 为什么沿 key 位置做, 以及第 0 个 token 为什么不能使用第 1 个 V。
核心原因：
1. Softmax 沿 Key 位置计算：因为每个 Query 需要决定自己应该关注哪些 Key，以及分配多少注意力权重。
2. 第 0 个 Token 不能使用第 1 个 V：因为 Causal Attention 不允许当前 Token 看到未来 Token 的信息。

