# week3 day2

本文回答 [第 3 周学习路线 Day2](../../003-llama-cpp-week3-daily-learning-route.md) 的当天输出与验收: 三类映射表、Q weight 从枚举到 GGUF 名称的过程、metadata 与 tensor 命名规则的区别, 以及权重名与 graph 激活名的区别。

核对基准为当前工作区 `dd545f7e7`, 与学习路线记录的 `545ddb30a` 不同。本文依据源码静态阅读和已有学习记录整理; 涉及重新读取 GGUF、运行模型或 GDB/callback 观察的手动实验均暂时跳过, 具体状态见文末。文中的调用示例用于解释源码, 本次未编译执行。

沿用 Day1 的符号: `D` 为 hidden size, `H` 为 Q head 数, `KvH` 为 KV head 数, `Dh` 为 head dimension, `I` 为 FFN 中间宽度, `N` 为当前输入 token 数。本周普通 dense Llama 假定 `D=H*Dh`, Q/K/V 的 head dimension 相同。逻辑权重写成 `[out,in]`, GGML 二维权重的 `ne[]` 写成 `[in,out]`, 省略尾部的两个 1。

## 习题1: 三类映射表

### 1. architecture: 架构字符串与枚举的双向映射

`general.architecture` 是 GGUF metadata 的 key, 它的字符串 value 才是这里要映射的架构名。不要把 key `general.architecture` 与 value `llama` 当成同一层概念。

| GGUF metadata | 字符串到枚举 | 枚举到字符串 | 模型工厂选择的类型 |
| --- | --- | --- | --- |
| `general.architecture = "llama"` | `llm_arch_from_string("llama") -> LLM_ARCH_LLAMA` | `llm_arch_name(LLM_ARCH_LLAMA) -> "llama"` | `llama_model_llama` |
| `general.architecture = "qwen2"` | `llm_arch_from_string("qwen2") -> LLM_ARCH_QWEN2` | `llm_arch_name(LLM_ARCH_QWEN2) -> "qwen2"` | `llama_model_qwen2` |

这两行的依据是 `llama-arch.h` 中的 `llm_arch` 和 `llama-arch.cpp` 中的 `LLM_ARCH_NAMES`。正向函数遍历该映射表, 比较字符串并返回枚举; 反向函数用枚举查出字符串。字符串比较按源码精确匹配, 不能把 `"Llama"` 默认视为 `"llama"`。未匹配的架构字符串返回 `LLM_ARCH_UNKNOWN`, 普通模型创建入口会拒绝未知架构。

模型加载中的连接关系为:

```text
GGUF metadata: general.architecture 的字符串 value
  -> llama_model_loader 读取到 arch_name
  -> llm_arch_from_string(arch_name)
  -> loader 保存 LLM_KV(对应的架构枚举)
  -> llama_model_create() 取得 ml.get_arch()
  -> llama_model_mapping() 选择模型子类
```

因此 GGUF 文件名中的品牌或模型系列名不决定 C++ 架构分支。学习路线沿用第 2 周记录, 将本地 `deepseek-r1-1.5b-f16.gguf` 作为 Qwen2 模型讨论; 本次确认的是 `qwen2` 的源码映射, 没有重新读取文件来实测 `general.architecture` 的 value。

### 2. metadata key: 配置枚举、格式字符串与实际 key

以 `LLM_KV kv(LLM_ARCH_LLAMA)` 为例, 默认不传 suffix。`kv(...)` 先从 `LLM_KV_NAMES` 取格式字符串, 再用架构名 `"llama"` 替换其中的 `%s`。它生成的是 key, 具体配置值要由 loader 再从 metadata 读取。

| 配置含义 | `llm_kv` 枚举 | `LLM_KV_NAMES` 格式字符串 | Llama 实际 key | 读取后的字段 / 符号 | 对公式或 shape 的影响 |
| --- | --- | --- | --- | --- | --- |
| hidden size | `LLM_KV_EMBEDDING_LENGTH` | `%s.embedding_length` | `llama.embedding_length` | `hparams.n_embd`, D | hidden 激活 `[N,D]`; Q/K/V、gate/up 权重的输入维度为 D |
| Q head 数 | `LLM_KV_ATTENTION_HEAD_COUNT` | `%s.attention.head_count` | `llama.attention.head_count` | `hparams.n_head_arr`, `n_head(il)`, H | Q 拆 head 为 `[N,H,Dh]`; 本周假设下 `Dh=D/H` |
| KV head 数 | `LLM_KV_ATTENTION_HEAD_COUNT_KV` | `%s.attention.head_count_kv` | `llama.attention.head_count_kv` | `hparams.n_head_kv_arr`, `n_head_kv(il)`, KvH | K/V 投影宽度为 `KvH*Dh`; 每组 `H/KvH` 个 Q heads 共用一组 K/V |
| FFN 中间宽度 | `LLM_KV_FEED_FORWARD_LENGTH` | `%s.feed_forward_length` | `llama.feed_forward_length` | `hparams.n_ff_arr`, `n_ff(il)`, I | gate/up 激活 `[N,I]`; down 投影从 I 恢复为 D |
| RMSNorm epsilon | `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS` | `%s.attention.layer_norm_rms_epsilon` | `llama.attention.layer_norm_rms_epsilon` | `hparams.f_norm_rms_eps`, eps | 用于 `x / sqrt(mean(x^2)+eps) * gamma`, 影响数值, 不改变 shape |

前四项由 `llama_model::load_hparams()` 的公共路径读取, RMS epsilon 由 `llama_model_llama::load_arch_hparams()` 读取。当前代码支持将 head 数、KV head 数和 FFN 宽度读取到逐层数组中, 本周用统一的 H、KvH、I 描述普通等宽模型。

两个容易漏掉的细节:

- `head_count_kv` 是可选配置: 公共加载逻辑先令 KV head 数等于 Q head 数, 再尝试读取该 key。因此缺少该 key 不表示没有 K/V, 而是默认 `KvH=H`。
- epsilon 是配置标量, norm 的 `gamma` 是权重 tensor。例如 `output_norm.weight` 保存 gamma, 不能用它代替 `llama.attention.layer_norm_rms_epsilon`。

以 KV head 数为例, 完整的“找配置”过程是:

```text
LLM_KV_ATTENTION_HEAD_COUNT_KV
  -> LLM_KV_NAMES 中的 "%s.attention.head_count_kv"
  -> LLM_KV(LLM_ARCH_LLAMA) 用 "llama" 格式化
  -> "llama.attention.head_count_kv"
  -> loader 按这个 key 读取配置值
  -> hparams.n_head_kv_arr
  -> K/V 权重宽度、head reshape、GQA 分组关系
```

同一个枚举配上 `LLM_ARCH_QWEN2`, 得到的是 `qwen2.attention.head_count_kv`。这是同一配置语义在不同架构命名空间中的 key, 不是把 Llama 配置直接用于 Qwen2。

### 3. tensor name: 权重枚举、名称模板与 GGUF 名称

`LLM_TENSOR_NAMES` 记录 tensor 的基础名称模板; `LLM_TN` 接收 tensor 枚举、suffix、block id 等参数, 返回 `LLM_TN_IMPL` 描述对象。最终通过 `.str()` 或隐式转换为 `std::string` 生成名称。

下表中 block id 取 0, suffix 取 `"weight"`, 按独立 Q/K/V 的普通 dense Llama 路径解释。

| 权重角色 / 枚举 | 名称模板 | 最终 GGUF tensor 名称 | 模型字段 | 逻辑 shape | GGML `ne[]` |
| --- | --- | --- | --- | --- | --- |
| Wq / `LLM_TENSOR_ATTN_Q` | `blk.%d.attn_q` | `blk.0.attn_q.weight` | `layers[0].wq` | `[H*Dh,D]` | `[D,H*Dh]` |
| Wk / `LLM_TENSOR_ATTN_K` | `blk.%d.attn_k` | `blk.0.attn_k.weight` | `layers[0].wk` | `[KvH*Dh,D]` | `[D,KvH*Dh]` |
| Wv / `LLM_TENSOR_ATTN_V` | `blk.%d.attn_v` | `blk.0.attn_v.weight` | `layers[0].wv` | `[KvH*Dh,D]` | `[D,KvH*Dh]` |
| Wgate / `LLM_TENSOR_FFN_GATE` | `blk.%d.ffn_gate` | `blk.0.ffn_gate.weight` | `layers[0].ffn_gate` | `[I,D]` | `[D,I]` |
| Wdown / `LLM_TENSOR_FFN_DOWN` | `blk.%d.ffn_down` | `blk.0.ffn_down.weight` | `layers[0].ffn_down` | `[D,I]` | `[I,D]` |
| 最终 norm 的 gamma / `LLM_TENSOR_OUTPUT_NORM` | `output_norm` | `output_norm.weight` | `output_norm` | `[D]` | `[D]` |

名称模板本身不存储 D、H 等维度。表中的 shape 来自模型的张量声明及 hparams; loader 还会检查文件中的实际 tensor shape 是否符合预期。`.weight` 也不表示某种固定 dtype, 实际类型要查看 tensor 信息。

## 习题2: Q weight 从枚举到 GGUF 名称的完整过程

### 1. 名称如何逐步生成

以以下调用为例:

```cpp
const auto tn = LLM_TN(LLM_ARCH_LLAMA);
const auto q_name = tn(LLM_TENSOR_ATTN_Q, "weight", 0);
const std::string name = q_name.str();
```

逐步展开:

| 步骤 | 源码行为 | 本例中间结果 |
| --- | --- | --- |
| 1 | `LLM_TN` 保存传入的 arch | `arch=LLM_ARCH_LLAMA` |
| 2 | 调用带 suffix 的 `LLM_TN::operator()` | 参数为 `tensor=LLM_TENSOR_ATTN_Q`, `suffix="weight"`, `bid=0`, 默认 `xid=-1` |
| 3 | 返回 `LLM_TN_IMPL` | 保存上述参数, 此时是名称描述对象 |
| 4 | `.str()` 查 `LLM_TENSOR_NAMES` | 取到模板 `"blk.%d.attn_q"` |
| 5 | `format(template,bid,xid)` | 模板只有一个 `%d`, 用 `bid=0` 得到 `"blk.0.attn_q"`; 此模板不使用 xid |
| 6 | suffix 非空, 追加 `"."` 和 suffix | `"blk.0.attn_q.weight"` |
| 7 | 返回字符串 | 作为 loader 查找 GGUF 权重的名字 |

这里的 `0` 是从 0 开始的 block 索引, 所以 `blk.0` 表示第一层。suffix 传入 `"weight"`, 不需要自己写前导点。若把 suffix 改成 `"bias"`, 生成的就是 `blk.0.attn_q.bias`。

### 2. 名称如何接到真正的权重查找

在当前源码中, Llama 的 `load_arch_tensors()` 调用 `create_tensor_qkv()`。该 helper 先尝试合并 QKV 权重; 在没有采用合并权重的独立分支, Q 的声明是:

```cpp
layer.wq = create_tensor(
    tn(LLM_TENSOR_ATTN_Q, "weight", bid),
    {n_embd_, n_embd_q_}, flags);
```

代入本例 `bid=0`, `n_embd_=D`, `n_embd_q_=H*Dh`, 得到:

```text
LLM_TENSOR_ATTN_Q + suffix="weight" + bid=0
  -> LLM_TN_IMPL
  -> .str() = "blk.0.attn_q.weight"
  -> llama_model_base::create_tensor(...)
  -> llama_model_loader::create_tensor(...)
  -> check_tensor_dims(tn.str(), 预期 ne=[D,H*Dh], ...)
  -> get_tensor_meta("blk.0.attn_q.weight")
  -> 在 loader 的 weights_map 中按名称查找 tensor 信息
  -> 核对 shape, 创建对应权重 tensor
  -> 返回给 layers[0].wq
```

这是加载已有 GGUF 权重时的查找链。`LLM_TN_IMPL::str()` 只生成名字, 不读取权重数据, 不计算 Q, 也不向 GGUF 写入新 tensor。后续图构建中的 `build_qkv()` 才使用 `layers[0].wq` 与输入激活构造投影运算。

不能从这个命名例子推断所有模型文件一定都有独立的 `blk.0.attn_q.weight`: 当前 `create_tensor_qkv()` 支持优先采用 `blk.0.attn_qkv.weight`, 此时 Q/K/V 可以来自合并投影。本文追踪的是题目要求的独立 Q weight 路径。

### 3. 对 ffn_gate 和 output_norm 复做一次

| 调用 | 模板展开 | suffix 追加 | 最终名称 |
| --- | --- | --- | --- |
| `tn(LLM_TENSOR_FFN_GATE,"weight",0)` | `blk.%d.ffn_gate` 加 `bid=0` 得到 `blk.0.ffn_gate` | 追加 `.weight` | `blk.0.ffn_gate.weight` |
| `tn(LLM_TENSOR_OUTPUT_NORM,"weight")` | `output_norm` 没有层号占位符, 默认 `bid=-1,xid=-1` 不进入名称 | 追加 `.weight` | `output_norm.weight` |

`ffn_gate` 是每个 block 的权重, 所以名称包含 `blk.<层号>`。`output_norm` 是所有 block 之后的最终 norm, 所以没有 block 前缀; 它与每层的 `blk.0.ffn_norm.weight` 是不同的权重。

## 习题3: metadata key 为什么常带架构前缀, 而 tensor 名可以跨架构复用?

直接原因在两个名称生成函数的实现不同:

```text
LLM_KV::operator():
  format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch))
  有 suffix 时再追加 "." + suffix

LLM_TN_IMPL::str():
  format(LLM_TENSOR_NAMES.at(tensor), bid, xid)
  有 suffix 时再追加 "." + suffix
```

metadata 模板经常包含 `%s`, 用来填入架构名, 例如 `llama.embedding_length` 和 `qwen2.embedding_length`。但并非所有 metadata key 都有架构前缀: `general.architecture`、`general.name`、`tokenizer.ggml.model` 等模板本身没有 `%s`。

tensor 模板表达的是角色及其所在 block, 例如 `blk.%d.attn_q`。虽然 `LLM_TN_IMPL` 对象也保存了 arch, **当前 `.str()` 实现不使用 arch 来格式化名字**。因此同样的角色可以在 Llama 和 Qwen2 中生成相同的标准 tensor 名。

| 配置或权重 | Llama | Qwen2 |
| --- | --- | --- |
| 架构 metadata 的 value | `llama` | `qwen2` |
| hidden size key | `llama.embedding_length` | `qwen2.embedding_length` |
| KV head 数 key | `llama.attention.head_count_kv` | `qwen2.attention.head_count_kv` |
| 独立 Q weight 名 | `blk.0.attn_q.weight` | `blk.0.attn_q.weight` |
| dense FFN gate weight 名 | `blk.0.ffn_gate.weight` | `blk.0.ffn_gate.weight` |
| 最终 norm weight 名 | `output_norm.weight` | `output_norm.weight` |

这些相同名称可在两种模型的 `load_arch_tensors()` 及公共 `create_tensor_qkv()` 中核对。名称复用表示共享权重角色的命名规则, 不代表 tensor 数值、shape、bias、RoPE 或整个模型计算图完全相同。仅凭一个 `blk.0.attn_q.weight` 名字不能判断模型是 Llama 还是 Qwen2, 还要看 `general.architecture`。

三类映射及 graph 节点命名的职责边界如下:

| 类型 | 输入 | 输出示例 | 作用与边界 |
| --- | --- | --- | --- |
| architecture | metadata value `"llama"` | `LLM_ARCH_LLAMA` | 识别架构, 交给模型工厂选择实现; 不提供具体权重数据 |
| metadata key | `LLM_KV_ATTENTION_HEAD_COUNT_KV` + arch | `llama.attention.head_count_kv` | 定位配置项, 读取其 value; 不是权重名称 |
| tensor name | `LLM_TENSOR_ATTN_Q` + layer + suffix | `blk.0.attn_q.weight` | 定位权重 tensor; 名字不编码完整 shape 或数值 |
| graph 中间节点名 | `cb(cur,"Qcur",0)` 及命名 callback | 通常为 `Qcur-0` | 标注计算图中的中间结果, 便于观察; 不是 GGUF 权重查找规则 |

它们分别回答“采用哪种模型实现”“配置值从哪里读”“权重从哪里找”“正在观察哪个计算阶段”。替换其中一个名字不能自动完成其他三件事。

## 习题4: 区分 blk.0.attn_q.weight 与 Qcur-0

| 比较项 | `blk.0.attn_q.weight` | `Qcur-0` |
| --- | --- | --- |
| 含义 | 第一层 Q 投影的固定权重 Wq | 第一层 Q 激活的常见 graph 节点名 |
| 名字从哪里来 | `LLM_TENSOR_ATTN_Q` 经 `LLM_TN_IMPL::str()` 生成 | graph builder 调用 `cb(Qcur,"Qcur",0)`, 由命名 callback 设置 |
| 是否用于 GGUF 权重查找 | 是, 独立 Q weight 路径按此名查找 | 否, 本例是构图时给激活起的名字 |
| 数值依赖 | 来自模型权重, 推理时固定 | 由当前输入和权重计算; RoPE 后还依赖 position |
| 逻辑 shape | `[H*Dh,D]` | 投影后 `[N,H*Dh]`; 拆 head 后 `[N,H,Dh]` |
| GGML `ne[]` | `[D,H*Dh]` | 投影后 `[H*Dh,N]`; 拆 head / RoPE 后 `[Dh,H,N,1]` |
| 与 token 数 N 的关系 | shape 不含 N | shape 随本次 N 改变 |
| 在计算中的角色 | 作为投影的输入参数 | 作为投影、reshape 或 RoPE 等计算的结果, 继续送入 attention |

无 LoRA、无额外 scale、无 bias 的独立投影路径为:

```text
模型权重 Wq: layers[0].wq
  名称 = blk.0.attn_q.weight
  ne   = [D,H*Dh]

当前激活 Xn:
  ne   = [D,N]

build_lora_mm(Wq,Xn)
  -> ggml_mul_mat(ctx0,Wq,Xn)
  -> Qflat.ne = [H*Dh,N]
  -> cb(Qflat,"Qcur",0)
  -> 常见名称为 Qcur-0

数学含义: Qflat = Xn Wq^T
```

命名细节还要分清两个位置: `llm_graph_context::cb()` 本身负责调用 `cb_func`; 标准 context 提供的 `llama_context::graph_get_cb()` 在 `il>=0` 时用 `ggml_format_name(cur,"%s-%d",name,il)` 设置名称, 所以 `name="Qcur",il=0` 得到 `Qcur-0`。这里的命名发生在构图阶段, 不是执行完成后才命名, 更不表示激活数值已经算好。

`Qcur-0` 也不是唯一节点标识。当前代码在投影后、可选 bias 后、reshape 后以及 Llama 的 RoPE 后都可能调用同名 `cb()`。因此需要结合 `op`、`ne[]`、`src[]`、tensor 指针或图节点索引辨认阶段。例如 `MUL_MAT` 且 `ne=[H*Dh,N]` 对应投影, `RESHAPE` 或 `ROPE` 且 `ne=[Dh,H,N,1]` 对应后续阶段。graph 构图 callback 与执行时的 eval callback 也应分开理解。

## 手动实验与验收状态

本次完成的是源码层面的解答, 不把未执行的操作记为实测。

| 项目 | 状态 | 证据或说明 |
| --- | --- | --- |
| Llama / Qwen2 架构双向映射 | 已完成源码核对 | `LLM_ARCH_NAMES`、`llm_arch_from_string()`、`llm_arch_name()` 与模型工厂分支 |
| 五项 metadata 映射 | 已完成源码核对 | 已列出枚举、模板、Llama key、目标字段和公式/shape 影响 |
| tensor 命名与 Q weight 完整追踪 | 已完成源码核对 | 已追到 `.str()`、loader 名称查找和 shape 检查, 并补充合并 QKV 分支边界 |
| ffn_gate / output_norm 名称推导 | 已完成源码核对 | 已说明 block 权重与全局权重的命名差别 |
| 架构前缀与跨架构 tensor 名复用 | 已完成解释 | `LLM_KV::operator()` 与 `LLM_TN_IMPL::str()` 的格式化参数不同 |
| 权重名与激活节点名的区别 | 已完成解释 | 已区分来源、依赖、shape、构图命名和重复节点名 |
| 重新读取本地 GGUF 的 `general.architecture` value 和 tensor 清单 | 暂时跳过手动实验 | 按本次要求未运行 GGUF 检查工具, 未新增文件实测证据 |
| 运行模型, 用 GDB / eval callback 观察 `Qcur-0` | 暂时跳过手动实验 | 该项可留到后续运行时验证; 本文只记录源码推导的命名和 shape |

已有 [第 2 周 Day1 记录](../week2/01.%20week2-day1/01.%20week2-day1.md) 将本地模型标为 Qwen2, 并摘录了 `qwen2.*` metadata key 与权重 shape。该记录同时注明, 当时 `llama-gguf` 日志只打印了 key, 没有打印 `general.architecture` 的具体 value。因此本次可以引用其既有模型记录作为背景, 但不能声称已经重新读取并确认了该字段的值。后续补实验时应保留能同时显示 key 和 value 的证据。

## 源码核对入口

| 文件 | 建议定位的符号 / 内容 |
| --- | --- |
| [llama-arch.h](../../../../src/llama-arch.h) | `llm_arch`、`llm_kv`、`llm_tensor`、`LLM_KV`、`LLM_TN`、`LLM_TN_IMPL` |
| [llama-arch.cpp](../../../../src/llama-arch.cpp) | 三个名称映射表, 架构双向转换, metadata 和 tensor 名称格式化 |
| [llama-model.cpp](../../../../src/llama-model.cpp) | `llama_model_mapping()`、`llama_model::load_hparams()`、`llama_model_base::create_tensor_qkv()` |
| [llama-model-loader.cpp](../../../../src/llama-model-loader.cpp) | `general.architecture` 读取、`get_key()`、`create_tensor()`、`check_tensor_dims()`、`get_tensor_meta()` |
| [models/llama.cpp](../../../../src/models/llama.cpp) | Llama 的 `load_arch_hparams()`、`load_arch_tensors()` 和 RoPE 后的 `Qcur` 命名 |
| [models/qwen2.cpp](../../../../src/models/qwen2.cpp) | Qwen2 复用标准 tensor 名称的声明路径 |
| [llama-graph.cpp](../../../../src/llama-graph.cpp) | `build_qkv()`、`build_lora_mm()` 和构图 `cb()` |
| [llama-context.cpp](../../../../src/llama-context.cpp) | `graph_get_cb()` 中 `%s-%d` 的节点命名规则 |
