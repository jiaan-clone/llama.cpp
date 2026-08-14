# Transformer 概念桥接

本文解释 [llama.cpp 8 周学习计划](llama-cpp-8-week-learning-plan.md) 中“Transformer 概念桥接路线”涉及的概念。

这里的“Transformer 语言”不是编程语言, 而是一套描述模型数据流的共同词汇。学习源码时, 先说清楚数据是什么、shape 是什么、经过了什么计算, 再把它映射到 C++ 函数和 GGML op。

## 1. 统一符号

本文使用逻辑 shape。GGML 的 `ne[]` 维度顺序可能与下面的书写顺序不同, 看到源码或调试输出时必须以实际 tensor 元数据为准。

| 符号 | 含义 |
| --- | --- |
| `B` | batch size, 一次处理的样本数或序列数 |
| `T` | sequence length, 当前序列中的 token 数 |
| `D` | hidden size, 每个 token 的隐藏向量长度 |
| `H` | query attention head 数 |
| `Dh` | 每个 attention head 的维度, 通常为 `D / H` |
| `KvH` | key/value head 数, GQA 中通常小于 `H` |
| `I` | FFN 的中间维度, 通常大于 `D` |
| `V` | vocabulary size, 词表大小 |
| `L` | Transformer block 层数 |

## 2. 从文本到下一个 token

一个 decoder-only Transformer 的简化推理流程如下:

```text
text
  -> tokenizer
  -> token ids                         [B, T]
  -> embedding lookup                  [B, T, D]
  -> L 个 decoder blocks               [B, T, D]
  -> final RMSNorm                     [B, T, D]
  -> output projection                 [B, T, V]
  -> 选择最后一个位置的 logits
  -> sampler
  -> next token id
```

训练时可以同时计算多个位置的损失。生成时则把新 token 追加到序列末尾, 反复执行同一流程, 所以称为自回归生成。

## 3. Token、Token ID 与 Tokenizer

### 3.1 Token

Token 是模型处理的离散单位。它不一定等于一个字、一个词或一个字符, 可能是一个常见词片段、标点、空格或特殊标记。

例如, 一段文本可能被切分成:

```text
"Transformer works" -> ["Transform", "er", " works"]
```

实际切分方式由模型词表和 tokenizer 决定。

### 3.2 Token ID

Token ID 是 token 在词表中的整数编号。模型不直接读取字符串, 而是读取整数序列:

```text
tokens:   ["Transform", "er"]
token ids: [  1256,       87 ]
```

ID 本身没有大小关系。编号 100 并不比编号 50“更相似”或“更重要”。它只是查表索引。

### 3.3 Tokenizer

Tokenizer 完成文本与 token ID 序列之间的转换:

```text
text -> token ids
token ids -> token pieces -> text
```

在 llama.cpp 中, 重点观察 `llama_tokenize()`、`llama_token_to_piece()` 和 `src/llama-vocab.cpp`。特殊 token 还可能表示序列开始、序列结束、换行或控制消息。

## 4. Embedding 与 Hidden State

### 4.1 Embedding table

Embedding table 是一个可学习的矩阵, 逻辑 shape 为 `[V, D]`。给定一个 token ID, 模型取出对应的一行:

```text
embedding_table: [V, D]
token ids:       [B, T]
lookup result:   [B, T, D]
```

Embedding 的作用是把离散 ID 转换为可以进行矩阵计算的连续向量。

### 4.2 Hidden state

Hidden state 是模型在某一层对每个 token 的当前表示。进入第一层前, 它通常就是 embedding; 经过每个 decoder block 后, shape 通常仍为 `[B, T, D]`, 但内容越来越包含上下文信息。

不要把 hidden state 和 embedding 混为一谈:

- embedding 是初始查表结果;
- hidden state 是经过若干层计算后的上下文表示。

## 5. Decoder-only Transformer 与 Decoder Block

### 5.1 Decoder-only Transformer

Decoder-only 表示模型只使用 Transformer decoder 这一类结构, 通过 causal mask 保证当前位置不能读取未来 token。GPT、Llama 等自回归语言模型属于这一类。

它与 encoder-decoder Transformer 的区别在于: decoder-only 模型通常只处理一条不断增长的序列, 不额外接收 encoder 输出。

### 5.2 Decoder block

一个 Transformer 通常由 `L` 个相似的 block 堆叠而成。Llama 风格的 pre-norm block 可以先用下面的抽象理解:

```text
x1 = x + Attention(RMSNorm(x))
x2 = x1 + FFN(RMSNorm(x1))
```

不同模型可能调整 norm、attention 或 FFN 的细节, 因此源码中的实际顺序应以对应模型实现为准。

## 6. Residual Connection

Residual connection 将子层输出加回子层输入:

```text
y = x + sublayer(x)
```

要求 `x` 和 `sublayer(x)` shape 相同, 通常都是 `[B, T, D]`。它的主要作用是保留原始信息, 让很多层可以叠加而不必让每层都重新学习完整表示。

在 GGML graph 中, 它通常表现为 `ggml_add()` 或等价的加法 op。看到加法时, 要先判断它是不是 residual, 不要把所有加法都当成 bias。

## 7. RMSNorm
RMSNorm （Root Mean Square Layer Normalization） 是一种轻量级的归一化方法，由Shen 等人在2019年提出，用于替代经典的 LayerNorm
RMSNorm 对每个 token 的 hidden vector 做缩放归一化。简化公式为:

```text
rms(x) = sqrt(mean(x^2) + epsilon)
y      = weight * x / rms(x)
```

它只使用均方根进行缩放, 不像 LayerNorm 那样额外减去均值。`weight` 是模型学习到的逐维缩放参数, shape 通常为 `[D]`。

RMSNorm 的目的不是改变向量的语义方向, 而是控制数值尺度, 让后续 attention 和 FFN 更容易稳定计算。

## 8. 线性投影与 Q/K/V

### 8.1 线性投影

线性投影把 hidden state 乘以一个权重矩阵:

```text
Y = X W + bias
```

具体模型可能没有 bias, 或在 GGML 中以转置后的存储方式表示。学习时先关注逻辑输入输出维度, 不要先纠结内存布局。

### 8.2 Q、K、V

Attention 会从 hidden state 生成三类向量:

```text
Q = X Wq
K = X Wk
V = X Wv
```

可以用一个不严格但有用的直觉记忆:

- `Q` 是当前 token 想查询什么;
- `K` 是每个 token 提供的可匹配索引;
- `V` 是匹配成功后真正取回的内容。

对每个 attention head, 逻辑上可以把它们看成:

```text
Q: [B, H,   T, Dh]
K: [B, KvH, T, Dh]
V: [B, KvH, T, Dh]
```

如果没有 GQA, 通常 `KvH = H`。

## 9. Attention

### 9.1 Attention score

当前位置的 Q 与历史位置的 K 做点积, 得到相关性分数:

```text
scores = Q K^T
```

对于单个 head, 如果 Q 和 K 的 shape 是 `[T, Dh]`, score 矩阵的 shape 是 `[T, T]`。第一个 `T` 表示查询位置, 第二个 `T` 表示被查询的位置。

### 9.2 Scaling

点积的维度 `Dh` 越大, 数值波动可能越大, softmax 也会变得过于尖锐。因此 attention 通常使用:

```text
scores = Q K^T / sqrt(Dh)
```

这一步叫 scaling, 目的是控制 score 的数值尺度, 不是改变可见位置。

### 9.3 Softmax

Softmax 把一行 score 转为和为 1 的权重分布:

```text
attention_weights = softmax(scores)
```

权重越大, 表示当前位置越关注对应的历史位置。softmax 之前通常先应用 causal mask。

### 9.4 加权 Value

最后用 attention 权重对 V 做加权求和:

```text
attention_output = attention_weights V
```

因此 attention 的核心不是“找出一个最相关 token”, 而是按照一组权重混合多个位置的 Value。

## 10. Causal Mask

Causal mask 规定每个位置可以看哪些位置。对于自回归模型, 位置 `t` 只能看 `0..t`, 不能看 `t+1` 之后的 token。

逻辑上的 mask 类似下三角矩阵:

```text
position 0: [1, 0, 0]
position 1: [1, 1, 0]
position 2: [1, 1, 1]
```

被屏蔽的位置通常在 score 上加一个极小值, 近似为 `-inf`, 这样 softmax 后权重接近 0。

causal mask 解决的是“能不能看”的问题; RoPE 解决的是“位置信息如何进入 Q/K”的问题。两者不能互相替代。

## 11. Multi-Head Attention

Multi-head attention 将 hidden dimension `D` 切分成多个 head, 每个 head 有自己的 Q/K/V 投影和注意力分布:

```text
[B, T, D]
  -> [B, H, T, Dh]
  -> 每个 head 独立 attention
  -> 拼接回 [B, T, D]
  -> output projection
```

不同 head 可以学习不同的关系, 例如局部邻近、长距离依赖或格式结构。实际实现可能融合 transpose、reshape 和矩阵乘法, 但逻辑数据流仍可以按上面理解。

## 12. RoPE

RoPE 是 Rotary Position Embedding 的缩写。它通过按照位置对 Q、K 的二维分量对进行旋转, 把位置信息加入 attention。

关键点:

- 主要作用在 Q 和 K, 不直接作用在 V;
- 不需要单独把一个位置向量加到 hidden state 上;
- 旋转角度随 position 和维度变化;
- Q 与 K 经过相同规则旋转后, 点积能够反映相对位置信息。

在 llama.cpp/GGML 中, 重点观察 `ggml_rope()` 以及模型 graph 中 RoPE 调用的位置。不要把 RoPE 和 causal mask 混为一谈。

## 13. GQA

GQA 是 Grouped-Query Attention 的缩写。它保留较多的 Query heads, 但让多个 Query heads 共享较少的 Key/Value heads:

```text
H   = query head count
KvH = key/value head count
H   >= KvH
```

当 `H` 是 `KvH` 的整数倍时, 每个 K/V head 可以服务一组 Q heads。这样可以减少 K/V cache 的大小和生成阶段读取 K/V 的带宽, 但不会把 Q head 数减少到同样的数量。

在源码中, 重点看 `n_head_kv`、`n_embd_k_gqa()`、`n_embd_v_gqa()` 以及 memory/KV cache 的维度。

## 14. FFN

FFN 是 Feed-Forward Network 的缩写, 对每个 token 的 hidden vector 独立处理, 不直接在 token 之间建立关系。典型逻辑是:

```text
[B, T, D]
  -> projection to [B, T, I]
  -> activation or gating
  -> projection back to [B, T, D]
```

很多 Llama 类架构使用 gated FFN, 常见形式包含 `gate`、`up` 和 `down` 三类权重, 激活函数常见为 SiLU。学习时先掌握“扩展维度 -> 非线性/门控 -> 压回 hidden size”, 再阅读具体权重布局。

FFN 与 attention 的分工不同:

- attention 在 token 位置之间交换信息;
- FFN 对每个位置做独立的特征变换。

在 llama.cpp 中, 重点观察 `build_ffn()` 和模型层中 FFN 权重的加载与调用。

## 15. Output Norm、Logits 与 Vocabulary

### 15.1 Output Norm

所有 decoder blocks 完成后, 模型通常再做一次 RMSNorm, 得到最终 hidden state:

```text
[B, T, D] -> final RMSNorm -> [B, T, D]
```

### 15.2 Output projection

Output projection 把 hidden vector 映射到词表大小:

```text
[B, T, D] -> [B, T, V]
```

最后一维的每个数对应一个词表 token。

### 15.3 Logits

Logits 是模型对每个候选 token 给出的未归一化分数。它们不是概率, 也不保证在 0 到 1 之间, 更不要求总和为 1。

常见的下一步是:

```text
logits -> temperature/top-k/top-p/grammar 等处理 -> selected token
```

在 llama.cpp 中, 可以从 `llama_get_logits_ith()`、`llama_sampler_apply()` 和 `llama_sampler_sample()` 观察这条路径。

## 16. Greedy、Random Sampling 与自回归生成

### 16.1 Greedy

Greedy 直接选择 logits 最大的 token:

```text
next_token = argmax(logits)
```

它简单且可复现, 适合调试和正确性对比, 但可能产生重复或过于保守的结果。

### 16.2 Random sampling

随机采样先对 logits 做过滤或缩放, 再从剩余候选的概率分布中抽样。temperature、top-k、top-p 和 grammar 都可能影响候选集合或概率。

采样通常位于主 Transformer graph 之外: graph 负责产生 logits, sampler 负责决定下一个 token。

### 16.3 Autoregressive generation

自回归生成把刚采样出的 token 追加到序列, 再作为下一步输入:

```text
context -> logits -> sample token t1
context + t1 -> logits -> sample token t2
context + t1 + t2 -> ...
```

这解释了为什么生成阶段通常一次只新增一个 token, 也解释了 KV cache 为什么重要。

## 17. Prefill 与 Decode

### 17.1 Prefill

Prefill 是处理初始 prompt 的阶段。prompt 中的多个 token 可以在满足 causal mask 的前提下放入一个 batch 或多个 micro-batch, 一次性建立历史状态和首批 logits。

```text
prompt tokens: [t0, t1, t2, ...]
主要特征: token 数较多, 计算密集, 填充 KV cache
```

### 17.2 Decode

Decode 是 prompt 之后逐步生成新 token 的阶段。每一步通常只输入一个新 token, 但需要读取此前缓存的 K/V:

```text
new token: [tn]
主要特征: 新 token 少, 反复读取历史 memory, 常受内存带宽影响
```

在 llama.cpp 中, 两个阶段都可能经过 `llama_decode()`, 区别主要体现在输入 batch 的形态、已有 memory 和 graph shape。

## 18. KV Cache 与 Model Memory

### 18.1 KV Cache

每个 Transformer layer 都会为已经处理过的位置保存 K 和 V。下一次 decode 时, 新 token 的 Q 可以与历史 K 比较, 并使用历史 V, 无需重新计算历史 token 的 K/V。

KV cache 保存的是推理中间状态, 不是模型权重:

| 对象 | 保存内容 | 是否由模型文件提供 |
| --- | --- | --- |
| Model weights | embedding、Q/K/V、FFN 等参数 | 是 |
| KV cache | 已处理 token 的 K/V | 否, 推理时产生 |
| Logits | 当前输出位置的候选分数 | 否, 每次计算产生 |

### 18.2 Model memory

llama.cpp 使用更一般的 memory 抽象管理 sequence、slot、KV 或其他模型状态。普通 Transformer 常见的是 KV cache, 但 recurrent 或 hybrid 架构的状态不一定只有 K/V。

因此学习时要区分:

- `llama_memory_i` 是接口和状态管理抽象;
- KV cache 是其中一种具体的 memory 形式;
- `sequence id` 用于区分多个并行或分叉的序列。

## 19. Context、Position、Batch 与 UBatch

### 19.1 Context

Context 是一次推理运行的上下文窗口和运行时状态。`n_ctx` 表示可容纳的 token 位置上限, 不是模型文件大小, 也不是 batch size。

### 19.2 Position

Position 是 token 在序列中的位置编号。RoPE、causal mask 和 KV cache 都会使用 position, 但用途不同:

- RoPE 根据 position 计算旋转;
- causal mask 根据 position 判断可见范围;
- KV cache 根据 position 管理历史状态。

### 19.3 Batch

`llama_batch` 是公共 API 使用的输入描述, 可能包含 token、position、sequence id 和 logits 标记。一个 batch 可以包含多个 token 或多个序列。

### 19.4 UBatch

`llama_ubatch` 是内部处理用的 micro-batch。它将公共 batch 按 `n_batch`、`n_ubatch`、memory slot 和 backend 能力拆分, 以便构造和执行实际 graph。

可以用下面的问题检查理解:

```text
公共 batch 有多少 token?
被拆成多少个 ubatch?
每个 ubatch 使用哪些 sequence/position?
graph 是否因为 shape 或 backend assignment 改变而重建?
```

## 20. Transformer 概念与 llama.cpp 的对应关系

| Transformer 概念 | llama.cpp/GGML 观察位置 | 要确认的事实 |
| --- | --- | --- |
| Tokenizer | `llama_tokenize()`, `src/llama-vocab.cpp` | 文本如何变成 token ID, 特殊 token 如何处理 |
| Embedding | `src/models/llama.cpp` 的 token embedding | `[B, T]` 如何变成 `[B, T, D]` |
| RMSNorm/residual | `build_norm()`, `ggml_add()` | norm 和残差分别改变什么 |
| Q/K/V | `build_attn()`, attention projection weights | Q/K/V 的逻辑 shape 和权重来源 |
| RoPE/mask | `ggml_rope()`, attention mask | 位置编码与可见性约束分别在哪里生效 |
| GQA/KV cache | `n_head_kv`, `llama_memory_i`, KV cache | Q head 与 K/V head 如何共享, cache 如何按 sequence 管理 |
| FFN | `build_ffn()` 和模型层 FFN 权重 | hidden 如何扩展、门控并压回 `D` |
| Logits/sampler | `llama_get_logits_ith()`, `llama_sampler_sample()` | graph 输出如何变成下一个 token |
| Prefill/decode | `llama_decode()`, `process_ubatch()` | 多 token prompt 与单 token generation 的 batch 差异 |
| Graph/backend | `llama_model::build_graph()`, GGML scheduler/backend | graph 是如何声明的, 又由哪个 backend 执行 |

## 21. 小型 shape 追踪例子

使用一个仅用于学习的配置:

```text
B=1, T=3, D=4, H=2, Dh=2, KvH=1, I=8, V=8
```

一次逻辑追踪可以写成:

```text
token ids                         [1, 3]
embedding                         [1, 3, 4]
Q                                 [1, 2, 3, 2]
K/V                               [1, 1, 3, 2]
attention scores                  [1, 2, 3, 3]
attention output                  [1, 3, 4]
FFN intermediate                  [1, 3, 8]
FFN output                        [1, 3, 4]
logits                            [1, 3, 8]
```

这不是要求你在源码中看到完全相同的维度顺序。GGML 可能通过 transpose、reshape 或不同的内存布局表示同一个逻辑 tensor。验收重点是:

1. 每个输出的元素含义能说清楚。
2. 矩阵乘法的内维度能够对齐。
3. Attention 输出和 residual 相加前 shape 相同。
4. FFN 最终回到 `D`, output projection 最终到 `V`。

## 22. 常见混淆

| 容易混淆的概念 | 正确区分 |
| --- | --- |
| token 与 word | token 是 tokenizer 定义的单位, 不一定是完整单词 |
| token ID 与 embedding | ID 是整数索引, embedding 是查表得到的向量 |
| embedding 与 hidden state | embedding 是初始表示, hidden state 是经过网络后的表示 |
| logits 与 probability | logits 是未归一化分数, probability 需要经过 softmax 或采样处理 |
| causal mask 与 RoPE | mask 控制可见性, RoPE 注入位置信息 |
| GQA 与 quantization | GQA 改变 attention 的 head 组织, quantization 改变权重/激活的数据表示 |
| prefill 与 decode | prefill 处理已有 prompt, decode 逐步追加生成 token |
| model weights 与 KV cache | weights 是模型参数, KV cache 是本次推理的中间状态 |
| graph build 与 graph compute | build 声明 op 和依赖, compute 才实际执行 |
| batch 与 ubatch | batch 是公共输入描述, ubatch 是内部执行切分 |

## 23. 学习验收

完成本路线后, 应能不看源码回答:

- 一个 token ID 如何变成 `[D]` 向量?
- 一层 block 为什么需要两个 residual 分支?
- Q/K/V 的职责分别是什么?
- 为什么 score 要除以 `sqrt(Dh)`?
- causal mask 和 RoPE 的区别是什么?
- GQA 如何减少 KV cache 的开销?
- 为什么 FFN 扩展后又回到 `D`?
- logits 为什么还不是最终 token?
- prefill 和 decode 的 batch 为什么不同?
- KV cache 保存了什么, 它为什么不是模型权重?
- `llama_batch` 为什么还要拆成 `llama_ubatch`?

如果这些问题只能背定义, 但不能写出输入输出 shape, 说明概念还没有形成可用于源码阅读的模型。下一步应回到第 21 节的小型例子, 重新追踪一遍。
