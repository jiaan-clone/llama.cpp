# week3 day5

本文回答 [第 3 周学习路线 Day5](../../003-llama-cpp-week3-daily-learning-route.md) 的当天输出与验收: Attention shape 表、helper/op 对照、`ne[]/nb[]` 预测, 以及布局操作、两次 attention matmul 和融合操作的解释。

源码基准为当前工作区 `dd545f7e7`。主线为普通 dense Llama、单 sequence、单 KV stream、causal attention、非 Flash Attention, 省略 LoRA、bias、额外 scale/clamp 和 control vector。本文的 shape、stride、依赖均为源码推导; 手动运行实验暂时跳过。

沿用 `D=H*Dh`, Q/K/V 的 head dimension 均为 Dh, `H % KvH == 0`。N 是本次输入 token 数, S 是有效历史及当前 token 总数, Kcap 是 graph 使用的 KV 范围长度, O 是请求输出的位置数, I 是 FFN 宽度, V 是词表大小。逻辑权重为 `[out,in]`, GGML 二维 `ne[]` 为 `[in,out]`。

## 习题1: 非 Flash Attention 的 shape 与依赖

### 1. 从 build_attn 的 KV 重载展开

本文定位的是第一个参数为 `llm_graph_input_attn_kv *` 的 `build_attn()`。其普通路径可按下面的顺序阅读:

```text
Qcur / Kcur / Vcur: 已完成投影与拆 head, Q/K 已做 RoPE
  -> cpy_k / cpy_v: 构建当前 K/V 写入 cache 的节点
  -> get_k / get_v: 取得含历史及当前位置的 cache view
  -> build_attn_mha()
       -> q/k/v 的 permute, 按需调整 V 布局
       -> ggml_mul_mat(k,q): 原始点积
       -> ggml_soft_max_ext(...): scale + mask + softmax
       -> ggml_mul_mat(v,P): 概率加权 V
       -> permute + ggml_cont_2d: 拼接 heads
  -> build_lora_mm(wo,...): attention 输出投影
```

这描述构图和依赖关系, 不表示调用 `cpy_k()` 时已经执行了 cache 写入。helper 将相关写入节点显式加入 graph, 实际数据操作在执行图时完成。

当前普通 `llama_kv_cache::cpy_k()` 和 `cpy_v()` 最终调用的是 `ggml_set_rows()`, 对应 `SET_ROWS`, 不是根据函数名猜测的 `CPY`。`ggml_set_rows(a,b,c)` 的依赖顺序在当前源码中是 `src[0]=b`、`src[1]=c`、`src[2]=a`, 即当前值、写入索引、目标 cache; 返回结果是目标存储的 view。V 的转置存储分支还可能先做 reshape、cont 或 padding。

### 2. Attention shape 表

逻辑列用 S 表示有效 KV 内容; GGML 列使用 Kcap, 包括需要 mask 掉的无效位置。下表保留第四维的单 stream `1`。

| 阶段 | 逻辑含义 / shape | GGML `ne[0..3]` | 变化或依赖 |
| --- | --- | --- | --- |
| Q 刚拆 head | `[N,H,Dh]` | `[Dh,H,N,1]` | 独立投影后 reshape; RoPE 不改变 shape |
| K/V 刚拆 head | `[N,KvH,Dh]` | `[Dh,KvH,N,1]` | 当前输入的 K/V, 不是全部历史 |
| attention 的 q | 每个 Q head 有 N 个 query | `[Dh,N,H,1]` | `permute(...,0,2,1,3)` |
| attention 的 k | 有效 cache `[S,KvH,Dh]` | `[Dh,Kcap,KvH,1]` | `get_k()` 的 view 换轴后得到 |
| 原始 score | 每个 head 的 `Q K^T`, 汇总 `[H,N,S]` | `[Kcap,N,H,1]` | `ggml_mul_mat(k,q)`, 沿 Dh 收缩 |
| mask | `[N,S]` 的有效区域, 对 head 广播 | `[Kcap,N,1,1]` | 同时约束 causal 与无效位置 |
| 概率 P | `softmax(score/sqrt(Dh)+mask)` | `[Kcap,N,H,1]` | `SOFT_MAX`, 沿 `ne[0]` 归一化 |
| 与 P 相乘时的 v | 同一份 value, 按 key 轴组织点积 | `[Kcap,Dh,KvH,1]` | 取决于 cache 布局, 可能需要 transpose + cont |
| 加权 V | 每个 Q head 输出 `[N,Dh]` | `[Dh,N,H,1]` | `ggml_mul_mat(v,P)`, 沿 Kcap 收缩 |
| 加权 V 换轴后 | 为按 token 拼接 heads 做准备 | `[Dh,H,N,1]` | `permute(...,0,2,1,3)`, 仍可能不连续 |
| 拼接 heads | `[N,H*Dh]` | `[H*Dh,N,1,1]` | `ggml_cont_2d()` 物化为连续输出 |
| Wo 投影 | `[N,D]` | `[D,N,1,1]` | `ggml_mul_mat(wo,cur)`, 送入第一次 residual |

V 有两条需要区分的布局路径:

```text
cache 中 V 未转置:
  get_v: [Dh,KvH,Kcap,1]
  -> permute: [Dh,Kcap,KvH,1]
  -> transpose + cont: [Kcap,Dh,KvH,1]

cache 中 V 已转置:
  get_v: [Kcap,KvH,Dh,1]
  -> permute: [Kcap,Dh,KvH,1]
  -> 已满足非 FA 的第二次 matmul 布局
```

`build_attn_mha()` 使用输入 V 的 `nb[1] > nb[2]` 判断转置存储状态。因此不能在所有日志中都强行寻找额外的 `v_cont` 节点。cache view 的 stride 还可能由实际 cache 容量决定, 不能仅凭 Kcap 推导所有 `nb[]`。

### 3. 为什么两次 matmul 的写法没有反过来?

对于 `ggml_mul_mat(ctx,a,b)`, 当前构图实现要求:

```text
a.ne[0] == b.ne[0]
b.ne[2] % a.ne[2] == 0
b.ne[3] % a.ne[3] == 0

result.ne = [a.ne[1], b.ne[1], b.ne[2], b.ne[3]]
result.src[0] = a
result.src[1] = b
```

共同的 `ne[0]` 是点积收缩轴, 还要满足实现对布局的要求。设 `j(h)=floor(h/(H/KvH))`, 用 GGML 轴索引写为:

```text
kq[s,n,h] = sum_d k[d,s,j(h)] * q[d,n,h]

z[d,n,h]  = sum_s v[s,d,j(h)] * P[s,n,h]
```

所以第一式在数学的 token 在前约定下正是 `Q K^T`, 第二式正是 `P V`。传参顺序来自 GGML 的索引约定, 不是把数学公式倒置。

维度代入为:

```text
ggml_mul_mat(k,q):
  [Dh,Kcap,KvH,1] 和 [Dh,N,H,1]
  -> [Kcap,N,H,1], 收缩 Dh

ggml_mul_mat(v,P):
  [Kcap,Dh,KvH,1] 和 [Kcap,N,H,1]
  -> [Dh,N,H,1], 收缩 Kcap
```

GQA 的分组共享对应 head 维度的整数倍关系。CPU 实现中 `r2=H/KvH`, `i02=i12/r2`, 正好是 Q head 到 KV head 的映射, 不必事先复制出 H 份 KV cache。

### 4. 对已有真实模型配置代入数值

引用 [第 2 周模型记录](../week2/01.%20week2-day1/01.%20week2-day1.md) 的 Qwen2 配置 `D=1536,H=12,KvH=2,Dh=128,I=8960,V=151936`。这只是既有配置的代入, 本次没有重新读取 GGUF 或运行模型, 也不是 Llama 实测。

| 位置 | 代入后的 `ne[]` |
| --- | --- |
| Q 拆 head | `[128,12,N,1]` |
| K/V 拆 head | `[128,2,N,1]` |
| q / k | `[128,N,12,1]` / `[128,Kcap,2,1]` |
| score / P | `[Kcap,N,12,1]` |
| matmul 使用的 v | `[Kcap,128,2,1]` |
| 加权 V / 拼接 heads | `[128,N,12,1]` / `[1536,N,1,1]` |
| Wo 输出 | `[1536,N,1,1]` |
| gate/up / down | `[8960,N,1,1]` / `[1536,N,1,1]` |
| 最终 logits | `[151936,O,1,1]` |

这里 `H/KvH=6`, Q heads 0-5 使用 KV head 0, 6-11 使用 KV head 1。N、O、Kcap 保留符号, 不凭 prompt 文本或配置中的 context length 猜运行值。原记录也注明 metadata 日志未直接打印所有 value, 本次仍按“历史记录代入”标注。

## 习题2: helper 展开为 GGML op

| 数学作用 | helper / API | 普通路径的 op 与依赖 | 关键说明 |
| --- | --- | --- | --- |
| embedding lookup | `build_inp_embd()` / `ggml_get_rows()` | `GET_ROWS`, embedding 表与 token ids | token 输入路径得到 `[D,N]` |
| RMSNorm | `build_norm()` | `RMS_NORM` 后接 `MUL`, 后者依赖归一化结果与 gamma | gamma 按特征缩放, 在 token 轴广播; 不减均值 |
| 线性层 | `build_lora_mm()` | 无 LoRA/额外 scale 时为 `MUL_MAT`, `src[0]=W`, `src[1]=X` | 名称含 LoRA 不代表一定启用 LoRA |
| 独立 Q/K/V | `build_qkv()` | 三次投影后分别 `RESHAPE` | 返回值已经拆 head |
| 合并 QKV | `build_qkv()` | 一次合并投影后用三次 `VIEW` | Q/K/V 从同一投影行按 offset 切出, 保留原始 token stride |
| 位置编码 | `ggml_rope_ext()` | `ROPE`, 依赖 Q 或 K、position, 可选 factors | Q/K shape 不变; V 不做 RoPE |
| 写入 KV cache | `cpy_k()/cpy_v()` | 当前普通 cache 最终为 `SET_ROWS`, 还可能带布局操作 | 源依赖顺序为当前值、indices、目标 cache |
| 读取 KV cache | `get_k()/get_v()` | `VIEW` | view 描述共享存储的范围和 stride |
| 换轴 | `ggml_permute()` | `PERMUTE`, `src[0]` 为原 tensor | 重排 `ne/nb`, 不复制元素 |
| 连续化 | `ggml_cont()/ggml_cont_2d()` | `CONT`, `src[0]` 为原 tensor | 执行时生成连续结果; 构图调用本身不搬数据 |
| QK 点积 | `ggml_mul_mat(k,q)` | `MUL_MAT`, `src[0]=k`, `src[1]=q` | 收缩 Dh |
| scale + mask + softmax | `ggml_soft_max_ext()` | `SOFT_MAX`, `src[0]=score`, `src[1]=mask`; scale 保存在 op 参数中 | 在 key 轴归一化 |
| 加权 V | `ggml_mul_mat(v,P)` | `MUL_MAT`, `src[0]=v`, `src[1]=P` | 收缩 Kcap |
| SwiGLU | `ggml_swiglu_split(gate,up)` | `GLU` 的 `SWIGLU` 子类型; `src[0]=gate`, `src[1]=up` | `SiLU(gate).*up`, 日志的 op 描述可能直接显示 `SWIGLU` |
| residual | `ggml_add()` | `ADD` | 两条支路对应同一批 token |
| 最后一层选行 | `ggml_get_rows()` | `GET_ROWS`, 激活与 out ids | 两条 residual 支路同步 `N -> O` |
| 收集图依赖 | `ggml_build_forward_expand()` | 递归登记已有节点, 本身不是数值 op | 不执行矩阵乘法 |

普通 dense FFN 的 helper 展开为:

```text
Rn [D,N]
  -> up   = MUL_MAT(Wup,Rn)       [I,N]
  -> gate = MUL_MAT(Wgate,Rn)     [I,N]
  -> T    = SWIGLU(gate,up)      [I,N]
  -> F    = MUL_MAT(Wdown,T)     [D,N]
```

`LLM_FFN_PAR` 表示 gate/up 都使用 Rn; 最后一层提前选行后将这些激活中的 N 换成 O, 权重 shape 不变。合并 QKV、optional bias、scale、clamp 等分支会改变节点组成, 不能把这张普通路径表当成所有配置的固定节点清单。

## 习题3: 区分 reshape、view、permute、cont

| 操作 | 是否共享原存储 | shape / stride 的处理 | 当前 API 的关键条件 |
| --- | --- | --- | --- |
| `ggml_reshape_3d()` | 是 | 按同一线性元素序列重新分组, 建立新 shape | 输入必须连续, 元素总数不变 |
| `ggml_view_3d()` | 是 | 调用者提供 shape、字节 stride 和 offset, 可表示子区域或带间隔的切片 | 范围和 stride 必须有效, 不能把不连续数据假装成连续数据 |
| `ggml_permute()` | 是 | 重排轴对应的 `ne[]` 与 `nb[]` | 不改变元素值或搬移数据; 重排后可能不连续 |
| `ggml_cont()` | 语义上产生新的连续结果 | 保持 shape, 重新按连续布局物化 | 数据处理发生在执行阶段 |
| `ggml_cont_2d()` | 语义上产生新的连续结果 | 同时指定二维 shape 并物化 | 元素总数必须一致; 是重新排布的计算, 不只是改元数据 |

`ggml_transpose()` 是交换前两轴的 view 操作, 同时交换相应 stride。不能把“转置/换轴”直接等同于“复制一个转置后的连续矩阵”。实际 backend 还可能优化执行, 上表描述的是图/API 语义。

### 一组完整的 ne[] / nb[] 预测

纸面设 `Dh=2,H=4,KvH=2,N=3,D=8`, 所有以下激活都是无额外 padding 的 F32, 每个元素 4 字节。该例不涉及真实 cache 容量, 不运行 GGML 程序。

| 阶段 | `ne[0..3]` | 预测 `nb[0..3]`, 单位为字节 | 含义 |
| --- | --- | --- | --- |
| Q 投影连续结果 | `[8,3,1,1]` | `[4,32,96,96]` | 每个 token 有 8 个连续特征 |
| `reshape_3d(Q,2,4,3)` | `[2,4,3,1]` | `[4,8,32,96]` | 拆为 Dh、H、N |
| `permute(Q,0,2,1,3)` | `[2,3,4,1]` | `[4,32,8,96]` | 交换 token/head 轴, 仍共享原存储 |
| 单独对上述 q 做 `cont` 的示意结果 | `[2,3,4,1]` | `[4,8,24,96]` | shape 相同, stride 不同; 此行不表示源码一定对 q 调用 cont |
| 加权 V 的连续输出 kqv | `[2,3,4,1]` | `[4,8,24,96]` | 第二次 matmul 输出 |
| kqv 换轴后 | `[2,4,3,1]` | `[4,24,8,96]` | head 轴不是可直接连续合并的布局 |
| `cont_2d(...,8,3)` 拼接 heads | `[8,3,1,1]` | `[4,32,96,96]` | 执行时将各 token 的 heads 重排为连续特征 |

普通未量化元素的字节偏移为:

```text
offset = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
```

例如 Q 中 `d=1,h=2,n=1` 的元素:

```text
换轴前坐标 [d,h,n,0] = [1,2,1,0]: offset = 1*4 + 2*8 + 1*32 = 52
换轴后坐标 [d,n,h,0] = [1,1,2,0]: offset = 1*4 + 1*32 + 2*8 = 52
```

地址未变, 说明 permute 只是换了索引解释。若创建新的连续 q, 同一语义元素会位于新缓冲区的 `1*4+1*8+2*24=60` 字节处, 数值需要由执行过程搬到新位置。

合并 QKV 还能说明 view 为什么不能随便替换成 reshape: 上例合并投影的 `ne=[16,3,1,1]`, 每个 token 行占 64 字节。切出的 Q view 为 `ne=[2,4,3,1]`, `nb=[4,8,64,192]`; K/V view 的 offset 分别为 32、48 字节。Q 每个 token 虽只占前 32 字节, 下一个 token 仍要跨 64 字节, 不能套用独立 Q 的 `nb[2]=32`。

量化类型按 block 存储, 不可将这里的 `nb[0]=4` 和逐元素公式直接套用到量化权重。cache view 也可能带容量相关 stride, 必须记录实际 type、`ne/nb`、`view_src/view_offs`。

## 习题4: 为什么实际图的节点可能比公式步骤少?

| 纸面公式 | 当前图中可能的表达 | 应如何解释 |
| --- | --- | --- |
| score 缩放、加 mask、softmax 三步 | 一个 `SOFT_MAX` 节点 | scale 在 op 参数中, mask 是输入依赖, 没有漏掉公式 |
| gate 的 SiLU、与 up 逐元素相乘 | 一个 `GLU/SWIGLU` 节点 | 直接表达 `SiLU(gate).*up` |
| `QK^T`、缩放/mask/softmax、加权 V | `FLASH_ATTN_EXT` | 数学含义仍是 attention; 未必有独立可观察的 score/P tensor |

这不意味着总节点数一定更少: reshape、view、cache 写入、选行、连续化等也会增加图节点。公式步数、graph node 数、callback 日志行数和 backend kernel 调用数是不同的计数对象。

## 选读: 当前 Qwen2 与 Llama 的源码差别

| 项目 | 当前源码事实 |
| --- | --- |
| Q/K/V bias | 两者均可经过公共 `create_tensor_qkv()` 和 `build_qkv()` 的可选 bias 分支, 真实文件中是否存在需另查 |
| RoPE 类型 | `llama_model_rope_type()` 将普通 Llama 映射为 `LLAMA_ROPE_TYPE_NORM`, Qwen2 映射为 `LLAMA_ROPE_TYPE_NEOX` |
| RoPE factors | Llama graph 调用 `model.get_rope_factors()`; 当前 Qwen2 graph 传入 `nullptr` |
| FFN | 两者普通路径均为 `LLM_FFN_SILU + LLM_FFN_PAR`; 当前 Llama graph 传可选 FFN bias/scale, Qwen2 对应参数传 `NULL` |

以上只完成静态对照, 不将“某架构一定有 bias”作为结论。

## 验收与手动实验状态

- Attention shape 表、两次 matmul 的收缩轴和 GQA 映射: 已完成源码推导。
- helper/op 对照, 含当前 KV 写入的 `SET_ROWS` 依赖: 已完成源码核对。
- `reshape/view/permute/cont` 的区别与 F32 `ne[]/nb[]` 预测: 已完成纸面推导。
- 既有 Qwen2 配置代入与三个融合例子: 已完成, 未当作新增实测。
- 实际 `nb[]`、cache 布局、bias 和运行时节点核对: 暂时跳过手动实验, 后续按 [Day6 记录](week3-day6.md) 补证据。

## 源码入口

- [llama-graph.cpp](../../../../src/llama-graph.cpp): `build_norm()`、`build_lora_mm()`、`build_qkv()`、`build_ffn()`、KV 版 `build_attn()` 和 `build_attn_mha()`。
- [llama-kv-cache.cpp](../../../../src/llama-kv-cache.cpp): `cpy_k/cpy_v/get_k/get_v` 的写入 op 和两种 V view 布局。
- [ggml.c](../../../../ggml/src/ggml.c): matmul、layout、softmax、GLU 和 `SET_ROWS` 的 `op/src/ne/nb` 设置。
- [CPU matmul](../../../../ggml/src/ggml-cpu/ggml-cpu.c): `r2`、`i02` 对应的 head 分组映射。
- [llama-model.cpp](../../../../src/llama-model.cpp)、[Llama graph](../../../../src/models/llama.cpp)、[Qwen2 graph](../../../../src/models/qwen2.cpp): RoPE 类型、factors 和 FFN 参数差别。

## 习题5 闭卷从 Q/K/V 写到 attention output, 标出发生转置或换轴的位置
下面以 llama.cpp 的 Dense LLaMA、单批次、支持 GQA 的 Self-Attention 为例，从 Q/K/V 一直推导到 Attention Output。

重点区分三种操作：

- `ggml_reshape_*`：改变 Tensor 的维度解释，不改变元素的线性顺序。
- `ggml_permute()` / `ggml_transpose()`：交换维度或轴，通常先产生不同 stride 的视图。
- `ggml_cont()`：将 Tensor 转为连续布局，必要时实际重排数据。

### 1. 定义维度

| 符号    | 含义                      |
| ----- | ----------------------- |
| `N`   | 当前 Query Token 数        |
| `T`   | 可访问的 KV Token 数（包含历史缓存） |
| `H`   | Query Head 数            |
| `Hkv` | KV Head 数               |
| `Dh`  | Q/K Head 维度             |
| `Dv`  | V Head 维度               |
| `D`   | 模型隐藏维度                  |

对于常见的 LLaMA，`Dh = Dv`，但这里分开表示。

投影完成后的逻辑形状：

\\[ Q:[N,H D_h] \\]

\\[ K:[T,H\_{kv}D_h],\qquad V:[T,H\_{kv}D_v] \\]

其中，K/V 已包含 KV Cache 中的历史信息。

### 2. 完整的逻辑计算流程

Q / K / V

[N, H·Dh] / [T, Hkv·Dh] / [T, Hkv·Dv]

Reshape：拆分 Head

Q: [N,H,Dh]　K/V: [T,Hkv,Dh/Dv]

Permute：换轴&#x20;

Q: [H,N,Dh]　K/V: [Hkv,T,Dh/Dv]

Score = Q × Kᵀ / √Dh

逻辑结果 [H,N,T]

Causal Mask + Softmax

[H,N,T]，沿 T 归一化

Attention × V

[H,N,Dv]

Permute + Contiguous：换轴并合并 Heads

[N,H·Dv]

Wo 输出投影

最终 Attention Output [N,D]

这里的 GQA 意味着多个 Query Head 共享一个 KV Head。逻辑上可以把 K/V 看成按组广播到 H 个 Query Head，不一定实际复制数据。

### 3. 对应 llama.cpp 的 GGML `ne[]`

下面采用单条序列、普通 GQA 路径，暂不考虑特殊架构、KV Cache padding 和多 stream。

| 操作                        | GGML `ne[]`       | 是否换轴            |
| ------------------------- | ----------------- | --------------- |
| Q 拆 Head 后                | `{Dh, H, N, 1}`   | 否，Reshape       |
| K 拆 Head 后                | `{Dh, Hkv, T, 1}` | 否，Reshape       |
| V 拆 Head 后                | `{Dv, Hkv, T, 1}` | 否，Reshape       |
| Q `permute(0,2,1,3)`      | `{Dh, N, H, 1}`   | 是               |
| K `permute(0,2,1,3)`      | `{Dh, T, Hkv, 1}` | 是               |
| V `permute(0,2,1,3)`      | `{Dv, T, Hkv, 1}` | 是               |
| `ggml_mul_mat(k,q)`       | `{T, N, H, 1}`    | K 被隐式转置参与乘法     |
| Mask + Softmax            | `{T, N, H, 1}`    | 否               |
| V `transpose`             | `{T, Dv, Hkv, 1}` | 是               |
| `ggml_mul_mat(v,kq)`      | `{Dv, N, H, 1}`   | V 作为左操作数被隐式转置   |
| Output `permute(0,2,1,3)` | `{Dv, H, N, 1}`   | 是               |
| 合并 Heads                  | `{H×Dv, N, 1, 1}` | Contiguous + 合并 |
| Wo 输出投影                   | `{D, N, 1, 1}`    | 否               |

其中 GQA 的矩阵乘法按 Query Head 到 KV Head 的分组关系进行广播。

### 4. llama.cpp 源码中的关键步骤

在 `src/llama-graph.cpp` 的 `build_attn_mha()` 中，非 Flash Attention 分支的核心代码可以简化为：

```
// 1. Q/K/V 换轴
q = ggml_permute(ctx0, q, 0, 2, 1, 3);
k = ggml_permute(ctx0, k, 0, 2, 1, 3);
v = ggml_permute(ctx0, v, 0, 2, 1, 3);

// 2. Score：逻辑上 Q @ K^T
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

// 3. Scale + Mask + Softmax
kq = ggml_soft_max_ext(
    ctx0, kq, kq_mask, kq_scale, 0.0f
);

// 4. V 转置并确保连续（按布局条件执行）
v = ggml_cont(ctx0, ggml_transpose(ctx0, v));

// 5. Attention 权重乘 V
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);

// 6. 换轴：把 Head 移回内层
cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

// 7. 合并多个 Heads
cur = ggml_cont_2d(
    ctx0, cur,
    cur->ne[0] * cur->ne[1],
    cur->ne[2] * cur->ne[3]
);
```

这是为便于理解而提炼的代码；实际源码会根据 V 的 stride 决定是否转置，并处理其他特殊情况。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1



### 为什么 `ggml_mul_mat(k, q)` 对应 \\(QK^T\\)？

因为对于 GGML：

\\[ \operatorname{ggml\\\_mul\\\_mat}(A,B) \quad\Longleftrightarrow\quad BA^T \\]

这里 A、B 按 GGML 的二维矩阵解释。

因此：

\\[ \operatorname{ggml\\\_mul\\\_mat}(K,Q)=QK^T \\]

注意：这是矩阵乘法的运算语义，不代表源码必须先调用一次 `ggml_transpose(k)`。

### 5. Flash Attention 的区别

启用 Flash Attention 后，Score、Mask、Softmax、乘 V 被融合：

```
cur = ggml_flash_attn_ext(
    ctx0, q, k, v,
    kq_mask,
    kq_scale,
    0.0f,
    0.0f
);
```

输出通常为：

```
cur->ne = {Dv, H, N, 1};
```

再合并 Head：

```
cur = ggml_reshape_2d(
    ctx0, cur,
    cur->ne[0] * cur->ne[1],
    cur->ne[2] * cur->ne[3]
);
```

得到：

```
cur->ne = {H * Dv, N, 1, 1};
```

然后通过 `Wo` 输出投影：

```
out = ggml_mul_mat(ctx0, Wo, cur);
```

最终：

```
out->ne = {D, N, 1, 1};
```

源码参考：[llama.cpp — llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp?utm_source=chatgpt.com)

最重要的是区分以下三种转置：

1. 显式换轴：`ggml_permute()`、`ggml_transpose()`，改变 Tensor 的轴或 stride。
2. 隐式矩阵转置：`ggml_mul_mat(A,B)` 的数学语义为 \\(BA^T\\)，不需要显式创建 \\(A^T\\)。
3. Head 重排：Attention 结束后，将 `{Dv,N,H}` 换成 `{Dv,H,N}`，再合并 Head，恢复每个 Token 的完整特征向量。

这三点是理解 llama.cpp Attention 张量布局的关键。

