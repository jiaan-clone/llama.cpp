# week6 day1: 词表、Tokenizer 分派与 round-trip

本文回答[第 6 周学习路线](../../006-llama-cpp-week6-daily-learning-route.md)中 Day 1 的当天输出与验收。

核对基准为当前 checkout `a8741cc99dadd9bea67ad91c225b69ce772909f6`, 与路线标注的 `545ddb30a` 不同。本文完成源码分析与本地 GGUF 元数据的只读检查; tokenize/detokenize 实跑、构建及 GDB 等手动实验按要求暂时跳过。下文的预测不是实测结果。

## 1. 从 GGUF 到 tokenizer 的分派图

```text
GGUF metadata
  +-- general.architecture --------> 模型架构及后续计算图
  |
  +-- tokenizer.ggml.model --------> llama_vocab::impl::load()
  |                                   +-- llama -> SPM
  |                                   +-- gpt2 -> BPE
  |                                   +-- bert -> WPM
  |                                   +-- t5 -> UGM
  |                                   +-- rwkv / plamo2 -> 对应实现
  |
  +-- tokenizer.ggml.pre ----------> pre_type、预切分规则及相关默认行为
  +-- tokenizer.ggml.tokens -------> id_to_token / token_to_id
  +-- scores / token_type ---------> token 分数和属性
  +-- merges ---------------------> BPE merge ranks
  +-- BOS/EOS ID、add_bos/add_eos --> 特殊 token 配置
                                      |
                                      v
                         init_tokenizer(type)
                                      |
                         llm_tokenizer_bpe 等实现
                                      |
text -> tokenize(add_special, parse_special)
          -> 特殊 token 文本分段
          -> 对普通文本执行对应 tokenizer
          -> 按配置添加特殊 token
          -> token IDs
```

这是逻辑依赖图, 不是 `load()` 每一行的执行顺序。源码先创建 tokenizer, 后面还会读取或修正特殊 ID、属性和添加策略。`no_vocab/none` 是没有普通文本词表的特殊分支, 不适用于本周模型。

三种名称不能混用: `qwen2` 是这里的模型架构; `gpt2` 指定 BPE 词表/tokenizer 家族; `deepseek-r1-qwen` 是 pre-tokenizer 名称, 当前代码将其映射到 `LLAMA_VOCAB_PRE_TYPE_QWEN2`。不能因为模型文件名带 DeepSeek 就使用另一个 DeepSeek 模型的 tokenizer 配置。

### 本地模型的静态元数据

直接读取 `/home/king/llama.cpp/deepseek-r1-1.5b-f16.gguf` 的 GGUF metadata, 未加载权重进行推理。

| 字段 | 当前文件中的值 | 含义 |
| --- | --- | --- |
| `general.architecture` | `qwen2` | 架构 |
| `tokenizer.ggml.model` | `gpt2` | `LLAMA_VOCAB_TYPE_BPE` |
| `tokenizer.ggml.pre` | `deepseek-r1-qwen` | Qwen2 预切分分支 |
| `tokenizer.ggml.tokens` 数量 | 151936 | ID 空间大小, 不是 embedding 维度 |
| `tokenizer.ggml.merges` 数量 | 151387 | BPE 合并规则数量 |
| `bos_token_id` | 151646 | `<｜begin▁of▁sentence｜>` |
| `eos_token_id` | 151643 | `<｜end▁of▁sentence｜>` |
| `padding_token_id` | 151643 | 此文件中与 EOS 相同 |
| `add_bos_token` | `true` | 允许自动添加特殊 token 时添加 BOS |
| `add_eos_token` | `false` | 不自动补 EOS |

表中的 token 文本保留原始 Unicode 字符, 不能把全角竖线替换为 ASCII `|`。上述 ID 只适用于当前模型。终止判定仍应使用 `llama_vocab_is_eog()`: 加载器会建立 EOG 集合, EOS 并不代表所有模型都只有一个结束 ID。

## 2. 文本、ID、embedding 和 piece 的关系

```text
输入文本 --tokenizer/词表--> token IDs --查 embedding 权重及模型计算--> logits
输出 IDs --token_to_piece/词表--> 字节片段 --按长度拼接--> 完整输出字节
```

tokenize 和 detokenize 都不需要运行 Transformer。`model_params.vocab_only=true` 适用于只研究词表的程序; 只有把 ID 送入模型推理的路径才需要 embedding 等权重。`llama_vocab_get_text()` 返回词表内部表示, BPE 的内部字符映射可能还需要解码, 不能直接把它当作输出字节。

单个 piece 是字节片段, 不保证独立构成 UTF-8 字符。例如汉字 `你` 的 UTF-8 是 `E4 BD A0`, 假设某词表将其拆成 `E4 BD` 和 `A0`, 分别打印可能出现替换符, 拼接后却是有效字符。这是字节边界示意, 不表示本地模型一定如此切分 `你`。

## 3. 参数与缓冲区契约

| API / 参数 | 控制内容 | 不能混淆的边界 |
| --- | --- | --- |
| `llama_tokenize(..., add_special, parse_special)` 的 `add_special` | 按模型配置自动添加 BOS/EOS 等 | `true` 不等于每个模型都同时添加 BOS 和 EOS |
| `parse_special` | 是否识别输入里的特殊/控制 token 文本 | 不负责自动添加 BOS; 当前实现即使为 false 仍会优先分割部分 USER_DEFINED token |
| `llama_token_to_piece(..., lstrip, special)` 的 `special` | 是否把控制/未知等特殊 token 渲染成文本 | 与编码时的 `parse_special` 是不同方向的开关 |
| `lstrip` | 最多跳过该 piece 开头指定数量的空格 | 不能对每个 piece 随意去空格, 否则改变原文 |
| `llama_detokenize(..., remove_special, unparse_special)` 的 `remove_special` | 按配置去掉序列边界的 BOS/EOS | 不是删除序列中所有特殊 ID; 实现也处理 tokenizer 的起始空格策略 |
| `unparse_special` | 是否显示剩余特殊 token 的文本 | 即使 `remove_special=false`, 也可能因它为 false 而不显示控制 token |

容量处理应遵守以下顺序:

1. tokenize 可先传输出容量 0 查询需求。若返回 0, 结果就是空 token 序列; 若返回 `INT32_MIN`, 按溢出失败处理; 其他负数 `-N` 才表示至少需要 N 个 token 槽位。
2. 分配后用完全相同的文本长度和开关重试, 检查返回值。`text_len` 是字节长度, 不是 Unicode 字符数。
3. token_to_piece 容量不足返回所需字节数的负值, 扩容后重试。成功时只追加返回的 N 个字节; 它不会补 `\0`, 不应直接当 C 字符串读取。
4. detokenize 同样以字节计数, 容量不足时扩容并重新调用。不要把第一次调用中的部分输出当成完整文本。

## 4. 六类 round-trip 对照表

统一以原始 UTF-8 字节比较。普通文本基线采用 `add_special=false, parse_special=false`, 单 piece 观察用 `lstrip=0`; detokenize 的两个开关必须逐行记录。对不含特殊文本的普通输入, 可用 `remove_special=false, unparse_special=false` 作基线。

下表是待执行矩阵及源码预测。实际普通文本 ID、piece 分界和字节差异未测, 不用猜测的 token ID 填充实验结果。

| 输入类别 | 固定输入或变化条件 | 当前可说明的结果 | 实测状态 |
| --- | --- | --- | --- |
| ASCII | `Hello llama.cpp`, 基线开关 | 15 个 UTF-8 字节; 不能据此推断 token 数 | ID、piece hex、恢复字节待测 |
| 中文混合 | `你好, llama.cpp`, 基线开关 | 17 个 UTF-8 字节、13 个 Unicode 字符; piece 可跨字符边界 | token 数及逐字节相等性待测 |
| 空白 | 字节 `20 20 41 20 20 42 0A 09 43 20`, 即两个前导空格、连续空格、LF、TAB、尾空格 | 用文件输入并关闭 CLI escape 转换; 记录首次差异字节偏移 | 待测 |
| 空串 | `add_special=false/true` | 本地 BPE 分支预测分别为 `[]` 与 `[151646]`; 后者配 `remove_special=true` 可去掉 BOS | 源码预测, 未调用 API |
| 特殊文本 | `<｜end▁of▁sentence｜>`, `add_special=false`, 切换 `parse_special` | true 时预测识别为 ID 151643; false 时按普通文本路径拆分。显示比较时需明确 `unparse_special` | 普通拆分 ID 与字节差异待测 |
| 自动 BOS | 同一普通文本, `add_special=false/true` | 本地配置预测 true 比 false 多一个前导 BOS, 不自动加 EOS; detokenize 是否移除要分别记录 | 待测 |

特殊文本测试可另做两种解码比较: `remove_special=false, unparse_special=true` 用于检查字面文本恢复; `unparse_special=false` 用于观察控制 token 被隐藏。后一种输出变短是参数要求的行为。

本地 BPE/Qwen2 分支关闭 `clean_spaces`, 普通文本预期按字节恢复; 仍需实际矩阵验证。对其他 tokenizer, normalization、空白清理、UNK 替代等可能使任意输入不再逐字节可逆。

### `tokenizer-roundtrip.csv` 应记录什么

计划位置为 `/home/king/llama-study/results/week6/tokenizer-roundtrip.csv`。因手动实验跳过, 本次不创建伪装成实测结果的 CSV。后续记录表头可为:

```csv
case,model,commit,input_hex,add_special,parse_special,remove_special,unparse_special,lstrip,token_ids,pieces_hex,output_hex,input_bytes,output_bytes,first_diff,status,explanation
```

`first_diff` 在相等时写明确的空值或约定值, `status` 区分 PASS、FAIL 和 SKIPPED。不要仅凭终端显示相同就判定字节相等。

## 5. 当天验收回答

**为什么中文 piece 显示异常未必是 tokenizer 错误?** UTF-8 字符可能分布在多个 piece 中。应按返回长度收集原始字节, 拼接后再进行 UTF-8 解码和原文比较。对单个 piece 使用字符串终止符或独立解码都可能造成误判。

**为什么重复添加 BOS 会改变模型输入?** 模板可能已经写入 BOS 文本, `parse_special=true` 将其转成 BOS ID, 同时 `add_special=true` 又自动添加一个。最终序列变成 `[BOS, BOS, ...]`, token 数、后续位置和 attention 的历史输入均变化。detokenize 隐藏 BOS 只能隐藏显示差异, 无法撤销模型已经接收的额外 token。BPE 分支会警告重复 BOS, 不会自动去重。Day 5 进一步说明 common 层移除模板 BOS、再由 tokenize 添加的路径。

**为什么 round-trip 必须声明比较条件?** 模型词表、原始字节、特殊 token 开关、首空格处理和 normalization 都影响结果。只有配置一致, 才能把不相等归因于需要调查的问题。

- [x] 已给出 GGUF 分派图、实际模型静态参数和 API 对照。
- [x] 已解释 piece/UTF-8、EOG、重复 BOS 和 round-trip 的比较前提。
- [ ] 六类输入实际 ID、piece 字节及恢复差异: 按要求暂时跳过。
- [ ] `tokenizer-roundtrip.csv` 实测文件及工具帮助/构建记录: 暂未生成, 不宣称动态验收通过。

源码入口: [llama-vocab.cpp](../../../../src/llama-vocab.cpp) 的 `impl::load/init_tokenizer/tokenizer_st_partition/tokenize/token_to_piece/detokenize`、[llama-vocab.h](../../../../src/llama-vocab.h)、[llama.h](../../../../include/llama.h) 的 Tokenization 区域、[tokenize.cpp](../../../../tools/tokenize/tokenize.cpp)。
