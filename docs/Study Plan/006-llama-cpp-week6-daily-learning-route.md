# llama.cpp 第 6 周每日学习路线: 词表、Tokenizer、Sampler、Grammar 和结构化输出

本路线根据 [llama.cpp 8 周深度学习与二次开发计划](llama-cpp-8-week-learning-plan.md) 的第 6 周安排展开, 承接 [第 5 周: KV Cache、序列管理、上下文移动和状态保存恢复](005-llama-cpp-week5-daily-learning-route.md)。本周把生成循环的两端接起来: 输入消息怎样变成 token, 输出 logits 怎样在历史、概率策略和语法约束下变成下一个 token。

本周主线:

```text
messages + model chat template
  -> common/Jinja -> formatted prompt
  -> vocab/tokenizer -> token IDs -> prefill/decode -> logits
  -> penalties / grammar / temperature / top-k / top-p
  -> greedy or dist -> selected token -> accept
  -> token pieces -> complete output bytes
  -> JSON parse -> schema validation -> application checks
```

完成标准: 能解释 tokenizer 的模型依赖、sampler 的状态与顺序、grammar 对候选 token 的约束方式; 完成一个不改变采样结果的观测 sampler, 并使 Planner 在约定配置下连续 20 次生成可解析且符合目标 Schema 的 JSON。

## 使用约定与本周边界

- 工作日每天 2 小时, 周末每天 4 小时, 共 18 小时。每天开头 10 分钟闭卷回忆, 结尾至少 15 分钟整理证据。
- 本文按工作区 `545ddb30a` 核对。总计划的基准版本较早, 以当前源码、`--help` 和 CMake target 为准, 用符号定位函数。
- 必做路径采用 CPU、host sampler、普通 decoder-only Transformer。关闭 backend sampling, 不涉及 speculative decoding、Mirostat、DRY、lazy grammar 或工具调用协议的完整实现。
- 沿用本地 `deepseek-r1-1.5b-f16.gguf`, 实际架构为 Qwen2。也可使用完整的本地 Llama/Instruct GGUF, 但同一组实验必须固定模型。词表、特殊 token 和 chat template 均从实际模型读取。
- 学习程序放在 `/home/king/llama-study/experiments/`, 复用第 1 周最小生成程序。先自行说明设计与断言, 再编写观测 sampler; 无需修改核心源码。遵守 [AGENTS.md](../../AGENTS.md) 与 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
- 本文规定待完成的任务, 不代表实验已经通过。20 次生成是第 7 天的验收目标, 不能由仓库单元测试替代。

开始前闭卷回答: logits 的长度为什么等于词表大小? `llama_get_logits_ith(ctx, -1)` 指向什么输出? 保存 context state 为什么不等于保存了 sampler RNG 和接受历史? 回答不清时先回看第 4、5 周。

## 7 天安排

| 天 | 时间 | 核心问题 | 当天主要输出 |
| --- | ---: | --- | --- |
| Day 1 | 2h | 文本、token ID、piece 和特殊 token 如何对应? | tokenizer 分派图、round-trip 表 |
| Day 2 | 2h | sampler 如何组合, 谁拥有状态, 谁调用 accept? | 生命周期图、两层 API 契约表 |
| Day 3 | 2h | 相同 logits 为什么会因采样顺序不同而得到不同分布? | 固定 logits 推演、候选变化表 |
| Day 4 | 2h | grammar 怎样约束 token, 怎样判断前缀与完整输出? | 枚举 grammar、正反例、断点记录 |
| Day 5 | 2h | 消息模板、输出 Schema、输出解析各负责什么? | prompt 构造图、Planner Schema 初稿 |
| Day 6 | 4h | 能否观察采样而不改变采样结果? | 观测 sampler、固定 logits 测试 |
| Day 7 | 4h | 怎样证明结构化输出完整、合法、可供应用使用? | 20 次生成报告、失败用例、周复盘 |

## 统一术语与记录表

| 名称 | 本周含义 | 必须区分的概念 |
| --- | --- | --- |
| vocabulary | token ID、文本、属性、分数及 tokenizer 配置等 | 词表不是模型的 embedding 权重 |
| tokenizer | 按模型配置把输入文本转换为 token ID | 一个汉字、一个单词和一个 token 不一一对应 |
| token piece | 一个 token 对应的输出字节片段 | 单个 piece 不一定是完整 UTF-8 字符 |
| logits | 当前预测位置上各 token 的未归一化分数 | 不是概率, 也不是 token ID |
| candidates | `llama_token_data_array` 当前有效候选视图 | `data` 指针、顺序和 `size` 都可能被 sampler 改变 |
| `selected` | 当前候选数组中的下标 | token ID 是 `data[selected].id` |
| `p` | 某一步计算出的概率字段 | 不保证每一步都已归一化或仍与当前 logits 一致 |
| `apply` | 处理当前候选, 可能过滤、改分数或选择 token | 随机选择也会推进 RNG, 不能视为纯函数 |
| `accept` | 把选定 token 提交给 sampler 的历史状态 | 不执行模型 decode, 不写 KV Cache |
| grammar | 约束可继续生成的 token 序列 | 语法约束不保证内容事实正确或任务可执行 |

采样记录至少包含: `step/stage/size/finite_logits/sorted/selected/selected_id/top_ids/entropy/seed/accepted_history`。其中 `finite_logits` 是未被屏蔽的有限 logit 数量, 不能只用 `size` 判断 grammar 的效果。

## Day 1: 词表加载、Tokenizer 分派与 round-trip

**投入: 2 小时。** 目标是解释本地模型的文本边界, 不要求实现一套 BPE 或读完全部 tokenizer。

### 学习路线

1. **0-10 分钟**: 画出 `text -> token IDs -> embeddings` 与 `token IDs -> pieces -> text`, 标出只有前一条推理路径需要模型权重计算。
2. **10-30 分钟**: 准备下方目录和工具, 保存版本、模型路径与帮助信息。已有构建可直接增量构建本周目标。
3. **30-55 分钟**: 阅读 [`src/llama-vocab.h`](../../src/llama-vocab.h)、[`src/llama-vocab.cpp`](../../src/llama-vocab.cpp) 的 `llama_vocab::impl::load()`、`init_tokenizer()` 和 `tokenize()`。追踪 `tokenizer.ggml.model`、pre-tokenizer、tokens、merges 和特殊 token 配置。区分模型架构名、tokenizer 类型、pre-tokenizer 类型。
4. **55-85 分钟**: 对照 [`include/llama.h`](../../include/llama.h) 的 `llama_tokenize()`、`llama_token_to_piece()`、`llama_detokenize()` 与 [`tools/tokenize/tokenize.cpp`](../../tools/tokenize/tokenize.cpp)。重点看容量不足的返回值和特殊 token 开关。
5. **85-105 分钟**: 在已有最小程序中增加独立的 tokenize/detokenize 实验, 填写下表。只需要词表时设置 `model_params.vocab_only=true`, 不运行 decode。
6. **105-120 分钟**: 整理差异原因, 回答为什么 tokenizer round-trip 需要明确比较条件。

### 环境与快速检查

```bash
cd /home/king/llama.cpp
export MODEL="$PWD/deepseek-r1-1.5b-f16.gguf"
export STUDY_DIR=/home/king/llama-study
mkdir -p "$STUDY_DIR/notes" "$STUDY_DIR/experiments" "$STUDY_DIR/results/week6"
git rev-parse HEAD > "$STUDY_DIR/results/week6/commit.txt"

cmake -S . -B build-study-debug \
  -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build-study-debug --target \
  llama-tokenize llama-completion test-sampling test-grammar-parser \
  test-grammar-integration test-gbnf-validator test-json-schema-to-grammar \
  test-jinja test-chat-template -j 4

./build-study-debug/bin/llama-tokenize --help > "$STUDY_DIR/results/week6/tokenize-help.txt"
./build-study-debug/bin/llama-completion --help > "$STUDY_DIR/results/week6/completion-help.txt"
./build-study-debug/bin/llama-tokenize -m "$MODEL" \
  --no-bos --no-parse-special -p 'Hello llama.cpp' --ids
```

若完整模型尚未准备好, Day 1 可先用 `models/ggml-vocab-qwen2.gguf` 跑词表实验; 这类文件不能用于后续模型生成。`llama-tokenize` 展示编码结果, 完整 round-trip 仍需调用 detokenize API。

### Round-trip 实验矩阵

| 输入 | 主要变化条件 | 记录内容 |
| --- | --- | --- |
| `Hello llama.cpp` | `add_special=false`, `parse_special=false` | ID、piece 字节、拼接输出 |
| `你好, llama.cpp` | 同上 | UTF-8 字节数、token 数, piece 是否跨字符边界 |
| 前导空格、连续空格、换行、制表符 | 使用文件输入并关闭 escape 转换 | 原始字节与恢复字节差异 |
| 空串 | `add_special=false/true` | 模型配置是否使结果出现 BOS/EOS |
| 实际模型中的一个特殊 token 文本 | `parse_special=false/true` | 普通文本拆分与特殊 ID 的区别 |
| 同一普通输入 | `add_special=false/true` | BOS/EOS 数量, detokenize 的移除策略 |

检查点:

- `add_special` 控制按模型配置自动添加特殊 token; `parse_special` 控制是否识别输入中的特殊 token 文本。二者互不替代。
- `llama_tokenize()` 可先查询容量再分配; 负数通常表示所需容量的负值, `INT32_MIN` 表示溢出, 不能无条件取负。
- `token_to_piece()` 不追加 `\0`, 按返回长度处理字节; 空间不足时扩容重试。`llama_vocab_get_text()` 的词表内部文本不等于解码后的 piece。
- 分别记录 `remove_special`、`unparse_special` 与 `lstrip`, 不把不一致的参数配置拿来证明 tokenizer 出错。存在 normalization 的 tokenizer 也不保证任意输入逐字节可逆。
- 用 `llama_vocab_is_eog()` 识别结束 token, 不假设只有一个 EOS ID。

### 当天输出与验收

- 从 GGUF 到 tokenizer 的分派图及参数对照。
- `results/week6/tokenizer-roundtrip.csv`: 至少六类输入的实际 ID、字节差异及解释。
- 能说明为什么中文 piece 的显示异常未必表示 tokenizer 错误, 以及为什么重复添加 BOS 会改变模型输入。

## Day 2: Sampler 接口、chain 和状态所有权

**投入: 2 小时。** 目标是把 sampler 当作有状态对象, 解释其生命周期和调用契约。

### 学习路线

1. **0-10 分钟**: 闭卷复述 Day 1 的四个特殊 token 开关。
2. **10-35 分钟**: 阅读 `include/llama.h` 的 `llama_sampler_i`、`llama_sampler`、chain API; 对照 [`src/llama-sampler.h`](../../src/llama-sampler.h) 与 [`src/llama-sampler.cpp`](../../src/llama-sampler.cpp)。画 `init -> apply -> accept -> reset/clone -> free` 图。
3. **35-65 分钟**: 跟踪 `llama_sampler_sample()` 与 chain 的 `apply/accept/free`, 标记候选数组构造、`selected` 到 token ID 的转换和所有权转移。
4. **65-90 分钟**: 对照 [`common/sampling.h`](../../common/sampling.h)、[`common/sampling.cpp`](../../common/sampling.cpp) 的 `common_sampler_sample()`、`common_sampler_accept()` 和 `common_sampler_reset()`。填写下面的契约表。
5. **90-105 分钟**: 用第 5 周 checkpoint 画出 `memory + pending input + sampler state + application history` 四列, 说明 reset、clone 和恢复历史各自能解决什么问题。
6. **105-120 分钟**: 写出三个容易造成错误的调用序列及其修正方式。

### 两层 API 的契约

| 路径 | 选择 token | 接受 token | 实验中应怎样调用 |
| --- | --- | --- | --- |
| `llama_sampler_apply()` | 取决于 chain 是否包含 greedy/dist 等选择器 | 不自动提交历史 | 检查 `selected`, 再显式 accept 一次 |
| `llama_sampler_sample()` 的 CPU 路径 | 获取 logits、apply、读取 `selected` | 内部已调用 `llama_sampler_accept()` | 返回后不再重复 accept |
| `common_sampler_sample()` | 执行 chain, 并处理 grammar 检查/必要的重采样 | 不自动调用 common accept | 用 `common_sampler_accept(..., true)` 接受生成 token |
| common 层接受 prompt | 不采样 | `is_generated=false`, 更新 chain 历史而不把 prompt 送入输出 grammar | 参考 completion 工具的 prompt 处理循环 |

上表限定 host sampling。当前源码存在 backend 已选出 token 后提前返回的路径, 不把 CPU 路径的结论推广到所有分支。

必须解释:

- `llama_sampler_chain_add()` 转移子 sampler 的所有权, 释放 chain 时释放子项; `chain_remove()` 后由调用者负责释放。不要把同一个子 sampler 交给两个 chain。
- `apply` 不等于无副作用查询: dist 会推进 RNG, 某些策略会维护内部状态。调试时额外调用一次 sample 可能改变后续输出。
- `accept` 会更新 penalties 的历史和 grammar 的解析状态。grammar 约束生成部分时, 不应把整段 prompt accept 到包含 grammar 的核心 chain。
- `reset` 的具体效果由子 sampler 实现决定; `clone` 应复制当前状态且保证之后独立演进。带非空 `ctx` 的自定义 sampler 若不实现 clone, 不能假设公共 API 会自动深拷贝。
- 重设相同 seed 是重新开始 RNG 序列, 不是恢复中途 RNG 状态。sampler clone 也不等于通用的磁盘序列化格式。

### 当天输出与验收

- 所有权图、两层 API 表、checkpoint 状态补充图。
- 能指出重复 accept、漏 accept、共享可变 sampler 分别会怎样污染历史或 grammar 状态。

## Day 3: 固定 logits、采样顺序与概率观察

**投入: 2 小时。** 先用四个候选推演, 再看真实词表; 避免仅靠生成文本判断 sampler 是否正确。

### 学习路线

1. **0-10 分钟**: 闭卷解释 `selected` 与 token ID 的区别。
2. **10-40 分钟**: 在 `src/llama-sampler.cpp` 阅读 penalties、top-k、top-p、temp、greedy、dist 的 `apply`。参考 [`tests/test-sampling.cpp`](../../tests/test-sampling.cpp) 的 `sampler_tester` 构造固定候选。
3. **40-70 分钟**: 完成下面的顺序实验, 每条路径均从同一份原始 logits 开始, 使用独立 sampler 状态。
4. **70-100 分钟**: 用 GDB 单步已有 `test-sampling`, 观察 penalties/top-k/top-p/temp/dist 的候选变化; 将实际结果补到下表。
5. **100-120 分钟**: 记录概率计算发生的位置, 解释为何不能在任意阶段直接用 `p` 算 entropy。

| 策略 | 主要行为 | 观察重点 |
| --- | --- | --- |
| penalties | 依据接受历史改变 logits | 正负 logits 的 repeat penalty 处理不同; frequency/presence 的计数来自 accept |
| top-k | 保留分数较高的候选 | `size`、候选 ID、排序与有效视图 |
| top-p | 计算概率并按累计质量截断 | `min_keep`、保留集合、截断后的概率和 |
| temperature | 正温度时按 `logit / T` 缩放 | 改变分布尖锐程度; `T <= 0` 的特殊分支 |
| greedy | 选择最大 logit 的候选下标 | 无需随机数; 仍不应把 `selected` 当 ID |
| dist | 从 logits 计算概率并随机选择 | `selected`、归一化概率、RNG 状态 |

### 一个可手算的顺序实验

使用四个不同 ID, 初始概率为 `[0.55, 0.25, 0.15, 0.05]`, logits 设为各概率的自然对数。设 `top_p=0.85`, `min_keep=1`, `T=0.5`。

- 路径 A: `top-p -> temperature -> dist`, top-p 初步保留前三项。
- 路径 B: `temperature -> top-p -> dist`, 温度变换后的概率约为 `[0.7756, 0.1603, 0.0577, 0.0064]`, top-p 初步保留前两项。
- 分别写出最终归一化概率与候选 ID。单次抽到相同 token 不能证明两个分布相同。
- 再给某个 token 接受两次历史, 应用 penalties, 观察排序变化。此时要从原始候选副本重新开始, 不把上一条路径截断后的候选继续当基线。

断点可按需启用:

```bash
gdb --args ./build-study-debug/bin/test-sampling
```

```gdb
set breakpoint pending on
break llama_sampler_penalties_apply
break llama_sampler_top_k_apply
break llama_sampler_top_p_apply
break llama_sampler_temp_apply
break llama_sampler_dist_apply
run
print *cur_p
print cur_p->data[0]
```

在函数返回后再次检查 `size/sorted/selected` 和当前 `data` 指针。`p` 可能来自较早阶段; 例如 top-p 截断后概率质量不一定为 1, temperature 改写 logits 后旧 `p` 也不再代表当前分布。

### 当天输出与验收

- `results/week6/sampler-stages.csv`: 同一 logits 在各阶段的有效视图和概率变化。
- 顺序实验的数值结果, 以及 greedy 与固定 seed 随机采样各自的复现条件。
- 能独立解释一个顺序变化会影响结果的例子, 同时说明正温度缩放通常不改变最大 logit 的 ID。

## Day 4: Grammar parser、候选屏蔽和前缀状态

**投入: 2 小时。** 先用有限枚举理解语法状态, 再解释 JSON grammar 的 token 级行为。

### 学习路线

1. **0-10 分钟**: 闭卷回忆 Day 3 的两条概率路径。
2. **10-35 分钟**: 阅读 [`grammars/README.md`](../../grammars/README.md)、[`src/llama-grammar.h`](../../src/llama-grammar.h) 与 [`src/llama-grammar.cpp`](../../src/llama-grammar.cpp) 的 parser、`llama_grammar_apply_impl()`、`llama_grammar_accept_impl()`。画出规则、解析栈集合和 `partial_utf8` 的关系。
3. **35-60 分钟**: 完成下方枚举 grammar 正反例, 区分可继续的前缀和完整匹配。
4. **60-90 分钟**: 用实际模型词表构造候选, 或在周末实验中设置 `llama_sampler_grammar_apply` 与 `llama_grammar_accept_impl` 断点。预测再观察非法候选、EOG 和多字节 piece 的处理。
5. **90-105 分钟**: 阅读 `common_sampler_sample()` 中 `grammar_first` 的两种路径, 画出先采样、检查单 token、必要时从原始 logits 重采样的分支。
6. **105-120 分钟**: 整理证据, 标出 grammar 和概率筛选之间的先后关系。

### 枚举 grammar 与完整输入验证

```bash
cat > "$STUDY_DIR/results/week6/decision.gbnf" <<'EOF'
root ::= "accept" | "revise"
EOF
printf '%s' 'accept' > "$STUDY_DIR/results/week6/decision-valid.txt"
printf '%s' 'ac' > "$STUDY_DIR/results/week6/decision-prefix.txt"
printf '%s' 'accept!' > "$STUDY_DIR/results/week6/decision-invalid.txt"

./build-study-debug/bin/test-gbnf-validator \
  "$STUDY_DIR/results/week6/decision.gbnf" "$STUDY_DIR/results/week6/decision-valid.txt"
./build-study-debug/bin/test-gbnf-validator \
  "$STUDY_DIR/results/week6/decision.gbnf" "$STUDY_DIR/results/week6/decision-prefix.txt"
./build-study-debug/bin/test-gbnf-validator \
  "$STUDY_DIR/results/week6/decision.gbnf" "$STUDY_DIR/results/week6/decision-invalid.txt"
```

预期: `accept` 完整合法; `ac` 可以继续生成, 但作为完整输入会报告意外结束; `accept!` 在追加 `!` 时非法。文件用 `printf` 写入, 避免多出的换行改变匹配结果。

当前 [`test-gbnf-validator.cpp`](../../tests/test-gbnf-validator.cpp) 即使报告输入非法也返回 0, 自动化断言必须检查其 valid/invalid 输出文本或直接调用验证逻辑, 不能只检查退出码。此工具验证字符级完整匹配, 不替代实际词表上的候选屏蔽、EOG 或跨 token UTF-8 测试。

关键结论:

- 普通字符 grammar 会检查 token piece 中的字符序列, 一个候选可以跨过多个 grammar 字符或产生尚未完整的 UTF-8 字符。当前 GBNF 还支持显式 token 规则, 不应概括为只支持字符匹配。
- `apply` 用当前状态检查候选, 将不允许的 logit 设为 `-INFINITY`; 通常不缩小 `size`, 不负责归一化概率, 不提交选中 token。
- `accept` 才推进已生成前缀的状态。存在可结束的解析状态时才允许 EOG; `stacks` 集合为空与集合中存在一个空 stack 具有不同含义。
- 自建实验使用 `penalties -> grammar -> temperature -> top-k -> top-p -> dist`, grammar 位于候选截断和最终归一化之前。若先 top-k 再 grammar, 可能提前丢掉全部合法 token。
- common 的默认路径可能先选择候选再做 grammar 检查, 非法时重建候选并先应用 grammar 后重采样。它没有直接接受非法 token; 但不能把此路径的 RNG 消耗或分布假定为与 `grammar_first=true` 完全相同。

### 当天输出与验收

- grammar 状态图、枚举正反例和 common 两条路径图。
- 能解释为什么合法前缀在 token 上限处被截断, 仍可能不是完整合法输出。

## Day 5: Jinja chat template、JSON Schema 与应用协议

**投入: 2 小时。** 把输入格式化和输出约束分别落到具体对象, 为第 8 周 Planner 协议做准备。

### 学习路线

1. **0-10 分钟**: 闭卷解释 grammar 的 apply/accept 与 EOG。
2. **10-40 分钟**: 阅读 [`common/chat.h`](../../common/chat.h)、[`common/chat.cpp`](../../common/chat.cpp) 的 `common_chat_templates_init/apply()`、`common_chat_templates_inputs`、`common_chat_params`, 再读 [`common/jinja/README.md`](../../common/jinja/README.md)。只跟踪当前模型实际使用的分支。
3. **40-65 分钟**: 参考 [`tests/test-chat-template.cpp`](../../tests/test-chat-template.cpp), 对同一组 user/assistant 消息比较 `add_generation_prompt=false/true` 的输出。保存完整 prompt 字节、特殊 token 和末尾生成前缀。
4. **65-95 分钟**: 阅读 [`common/json-schema-to-grammar.h`](../../common/json-schema-to-grammar.h)、[`common/json-schema-to-grammar.cpp`](../../common/json-schema-to-grammar.cpp) 与 [`tests/test-json-schema-to-grammar.cpp`](../../tests/test-json-schema-to-grammar.cpp), 用下方 Schema 生成并检查 GBNF。
5. **95-120 分钟**: 明确 Planner 的三个验证层和 reasoning 输出边界, 保存 Schema、prompt 和正反例草稿。

### 必须画出的两条路径

```text
messages + template -> common_chat_templates_apply -> prompt -> tokenize
schema -> json_schema_to_grammar -> grammar sampler -> allowed generated tokens
```

- 当前仓库有独立的 `common/jinja` 引擎, 路线是 lexer/parser/runtime。核心 `llama_chat_apply_template()` 只支持预定义模板, 不能替代 common 层的通用 Jinja 执行。
- `common_chat_params` 除 prompt 外还包含 grammar、generation prompt、parser 等信息。模板属于应用输入与输出协议层; 计算图只处理后续 token/embedding。
- 普通 `--json-schema-file` 转换不会自动把字段语义写进提示词。prompt 应明确输出字段、角色和任务, 并说明只输出 JSON。特殊模板/tool 路径可能另外处理 Schema, 要以当前分支为证据。
- 记录模板的输入来源标记与特殊 token 处理方式。完整格式化 prompt 只应用一次模板, 并检查 BOS 是否重复; 不把所有模型套进同一段手写角色分隔符。
- 本地 DeepSeek reasoning 模型的模板可能带 thinking 前缀。先检查实际 prompt 尾部及模板支持的参数; 本周选用从第一枚生成 token 起约束为纯 JSON 的实验协议, 记录这一选择对内容质量的影响。自动识别 reasoning 结束后再启动 lazy grammar 留作选做。

### Planner 的最小 Schema

把以下内容保存为 `results/week6/planner.schema.json`。字段有界, 便于 CPU 上完成短生成与失败测试。

```json
{
  "type": "object",
  "properties": {
    "goal": {"type": "string", "minLength": 1, "maxLength": 48},
    "steps": {
      "type": "array",
      "minItems": 1,
      "maxItems": 3,
      "items": {
        "type": "object",
        "properties": {
          "role": {"type": "string", "enum": ["worker", "reviewer"]},
          "task": {"type": "string", "minLength": 1, "maxLength": 64}
        },
        "required": ["role", "task"],
        "additionalProperties": false
      }
    }
  },
  "required": ["goal", "steps"],
  "additionalProperties": false
}
```

```bash
python3 examples/json_schema_to_grammar.py \
  "$STUDY_DIR/results/week6/planner.schema.json" \
  > "$STUDY_DIR/results/week6/planner.gbnf" \
  2> "$STUDY_DIR/results/week6/schema-conversion.stderr.txt"
```

检查转换警告和生成规则。C++ 路径使用 `json_schema_to_grammar(schema, true)` 可明确要求 GBNF; Python 转换是便于查看规则的独立入口, 不用一方通过来替代另一方验证。转换器支持 JSON Schema 的子集, 不把接受某个关键字等同于完整实现其约束。

三层验证分别是: JSON parser 能完整消费输出; Schema 校验字段、类型、枚举、长度与额外属性; 应用再判断任务是否相关、步骤是否合理。此 Schema 允许只有 reviewer 的计划, 因而“必须先 worker 后 reviewer”如有需要属于额外业务规则, 要单独记录。

### 当天输出与验收

- prompt 构造图、当前模型的模板与生成前缀说明。
- `planner.schema.json`、`planner.gbnf`、格式化后的 `planner-prompt.txt` 和转换日志。
- 能解释为什么合法 JSON 仍可能不满足 Schema, 满足 Schema 仍可能不能完成用户任务。

## Day 6: 实现不改变候选的观测 sampler

**投入: 4 小时。** 目标是保留一个小型可观测组件, 使用公共 sampler 接口, 不引入新的采样算法。

### 时间安排

| 时间段 | 任务 |
| --- | --- |
| 0-10 分钟 | 闭卷复述两层 sample/accept 契约 |
| 10-35 分钟 | 自行写出 observer 的输入输出、状态、所有权与断言 |
| 35-100 分钟 | 实现 observer 并接入固定 logits 程序 |
| 100-160 分钟 | 完成下方必要测试, 检查 clone/reset/free |
| 160-215 分钟 | 接入已有最小生成循环, 做 greedy 与随机链的短对照 |
| 215-240 分钟 | 整理结果, 为 Day 7 留出模型循环与输出文件接口 |

### 实现边界

通过 `llama_sampler_i` 实现一个 observer, `apply` 只记录当前候选快照, `accept` 记录最终接受的 token。它不负责选择 token, 不改 candidates, 不调用任何额外随机采样。基线与插入 observer 后的路径分别使用相同初始状态。

建议最小链:

```text
penalties -> grammar -> temperature -> top-k -> top-p -> observer -> dist
                                                           |
                                              snapshot before selection

llama_sampler_sample -> chain accept -> observer records accepted token
```

- 无 grammar 的固定 logits 测试可以省略 grammar 节点; greedy 对照用相同预处理加 greedy 选择器。
- `apply` 中记录 `size`、有限 logits 数、`sorted`、当前 `selected` 和 top candidates。排序只在拷贝上进行, 不保存指向临时候选的悬空指针。
- entropy 使用局部重新计算的概率: `q_i = exp(logit_i - max_logit) / sum(exp(...))`, `H = -sum(q_i * ln(q_i))`, 单位为 nat。屏蔽项概率为 0; 不直接依赖可能过期的 `p`。
- observer 位于 dist 前时, `selected` 通常尚未设置。通过后续 `accept` 记录最终 token, 不把某个 top candidate 当作已经选中的结果。
- 空候选、全部 `-INFINITY`、NaN 或正无穷需要独立诊断, 不计算伪 entropy。测试程序发现不可采样输入后停止, 不将其继续交给 dist 并期待恢复。
- observer 不更改候选的 `data/size/sorted/selected/id/logit/p`; 日志状态独立拥有。实现 reset、clone 和 free, 初始化未使用的 backend callbacks 为空, 确保 interface 生命周期覆盖 sampler。
- chain 拥有 observer; observer 的 free callback 释放自身 `ctx`, 外层 `llama_sampler` 由公共 free 释放。避免重复释放。

### 必须通过的测试

| 测试 | 输入/动作 | 断言 |
| --- | --- | --- |
| 观测透明性 | 同一候选经过 observer 前后 | 所有候选字段和数组元数据不变 |
| entropy | 两个等 logit、单一有效项、一个屏蔽项 | 分别得到约 `ln(2)`、0, 屏蔽项无贡献 |
| 选择不变 | 有/无 observer, 两个同 seed 的独立 chain | 每一步选中 ID 一致, 连续比较至少 32 步 |
| 接受历史 | 固定 logits, 显式接受重复 token | penalties 状态符合预期, observer 每个生成 token 只记录一次 accept |
| reset | 运行若干步后 reset, 与新建固定 seed 链对照 | 历史清空, 对本实验所用 sampler 结果符合其 reset 语义 |
| clone | 运行若干步后 clone, 向两边给相同候选和历史 | 初始续写一致, 单独推进其中一边不修改另一边状态 |
| 异常输入 | 空候选、全屏蔽、非有限异常值 | 诊断明确, 无越界、无伪造的概率结果 |

固定 logits 的连续步骤也要每轮重建候选, 并准确执行一次 accept。真实模型短对照从同一 prompt 与干净 memory 开始, 固定参数与线程; 用 token IDs 比较, 不只比较屏幕文本。

### 当天输出与验收

- `experiments/week6-sampler-study.cpp`: 小型 observer、固定 logits 测试和复用的生成入口。
- `results/week6/observer-tests.txt`、`observer-trace.csv`、有/无 observer 的 token 对照。
- 能逐行解释 observer 的状态与释放方式, 并证明它没有多消耗随机数或改动候选。

## Day 7: Planner 结构化生成、失败路径与周验收

**投入: 4 小时。** 把前六天的边界落到完整输出, 为第 8 周多 Agent 协议准备可复用资产。

### 时间安排

| 时间段 | 任务 |
| --- | --- |
| 0-10 分钟 | 闭卷复述 text/token/logits/grammar/JSON 的边界 |
| 10-40 分钟 | 完成 Schema 正反例和生成文件接口 |
| 40-140 分钟 | 连续生成 20 次, 保存原始输出、配置与终止原因 |
| 140-180 分钟 | 执行截断、非法 JSON、Schema 违例和状态重置测试 |
| 180-210 分钟 | 运行本周相关现有测试并记录证据范围 |
| 210-240 分钟 | 写报告, 口述验收, 整理交给第 8 周的文件 |

### 1. 固定协议与输出边界

复用 Day 5 的 Schema 和模板知识, 在 Day 6 的最小生成循环上完成实验。生成器从第一枚新生成 token 开始应用非 lazy grammar, 不把 prompt 喂入 grammar。每次请求重建 sampler, 并清空 memory、应用历史、输出字节和计数; model 可以只加载一次。

将新生成的非 EOG token pieces 按返回长度追加到独立输出文件; 诊断、token ID、采样统计写入另一文件。检测 EOG、token 预算、context 容量和 decode 错误, 保存终止原因。结构化结果需完整解析, 不截取第一个 `{...}` 来掩盖多余文本。

以下命令只用于快速观察 common 层的 Schema 约束, 不替代实验程序的结构化输出文件:

```bash
./build-study-debug/bin/llama-completion -m "$MODEL" -ngl 0 \
  -c 2048 -n 256 --seed 42 --temp 0.7 \
  --samplers 'temperature;top_k;top_p' --top-k 40 --top-p 0.9 \
  --no-conversation --no-display-prompt --simple-io \
  --json-schema-file "$STUDY_DIR/results/week6/planner.schema.json" \
  -p 'Return only JSON with goal and steps. Each step has role (worker or reviewer) and task. Use 1 to 3 short steps to review a C++ function.' \
  > "$STUDY_DIR/results/week6/planner-cli.stdout.txt" \
  2> "$STUDY_DIR/results/week6/planner-cli.stderr.txt"
```

此命令使用原始 completion prompt 来隔离采样行为; 正式实验使用 Day 5 核对过的模型模板。当前 completion 工具会输出 ` [end of text]` 等界面文本, 即使关闭 prompt 回显, stdout 也不能直接当作模型生成的 JSON。CLI common 路径和自建 grammar-first 链的顺序也应分别记录。

### 2. 连续 20 次生成矩阵

| 维度 | 固定安排 |
| --- | --- |
| 任务 | 5 个短任务: 学习安排、C++ 函数审查、单元测试计划、性能排查、文档整理 |
| 每个任务 | 1 次 greedy, 3 次随机采样, seed 分别为 11、22、33, 共 20 次 |
| 随机链 | `penalties -> grammar -> temperature -> top-k -> top-p -> observer -> dist`, 记录所有参数; 首轮可将 penalties 配为无作用 |
| 资源预算 | 固定 CPU/线程与 context; 起始最大生成 256 tokens, 先用一个短案例估算耗时和完整性 |
| 输入 | 保存实际格式化 prompt、模型标识、模板选项与 Schema |
| 输出 | 原始生成字节、token IDs、终止原因、三层校验结果、耗时 |

如果单次样例无法在预算内完整生成, 先调整任务长度或预算再冻结正式矩阵。失败运行必须保留; 修正配置后重新跑完整的连续 20 次, 不只挑选成功样本。

验收同时报告:

1. **完整生成**: 正常结束, 无 token/context 截断或 decode 错误。
2. **JSON 语法**: parser 完整消费输出, 除空白外无尾随内容。
3. **Schema**: required、类型、枚举、字符串长度、数组上下限和额外字段均通过独立校验。
4. **业务质量**: 计划与输入任务相关, 步骤可理解; 这一项单独计数, 不混入语法通过率。

Schema 校验可使用已有 JSON Schema 库; 若写针对本 Schema 的小型校验器, 必须覆盖所有用到的关键字, 字符串长度按字符而非 UTF-8 字节计数。不要仅用 `json.loads()` 或 grammar 转换成功来替代 Schema 校验。

### 3. 失败用例

| 用例 | 预期 |
| --- | --- |
| 将有效 JSON 截断在字符串或数组中间 | 完整解析失败, 不能算作合法前缀即成功 |
| 将 `steps` 改为字符串 | JSON 可解析, Schema 失败 |
| role 改为 `planner`, 或 steps 改为空数组 | Schema 失败 |
| 删除 goal, 添加额外字段, 超过长度上限 | Schema 失败 |
| 保留同样结构但令任务内容无关 | Schema 可能通过, 业务检查失败 |
| 故意设极小 token 预算 | 记录 budget stop, 本轮不计为完整成功 |
| 连续两个请求 | 第二个从干净 grammar/history/memory 开始, 不继承第一个的状态 |

### 4. 现有测试与证据范围

```bash
ctest --test-dir build-study-debug -N \
  -R '^test-(sampling|grammar-parser|grammar-integration|json-schema-to-grammar|jinja|chat-template)$'
ctest --test-dir build-study-debug --output-on-failure \
  -R '^test-(sampling|grammar-parser|grammar-integration|json-schema-to-grammar|jinja|chat-template)$' \
  > "$STUDY_DIR/results/week6/ctest.txt" 2>&1
```

先查看清单, 再运行测试。精确名称避免误选需要模型 fixture 的广泛测试或额外的 `test-jinja-py`。JSON Schema 测试会探测 `python` 命令是否提供 Python 3.8 或更高版本; 不满足时会跳过 Python 转换检查, 因此要检查日志中的实际覆盖范围。Day 5 的命令明确使用 `python3`。使用 CTest 可以保留其在 CMake 中配置的工作目录。

| 测试 | 能提供的证据 | 仍需自行验证 |
| --- | --- | --- |
| `test-sampling` | 内置采样策略的固定候选行为 | 自定义 observer 透明性与生命周期 |
| `test-grammar-parser` / `test-grammar-integration` | grammar 解析及已有匹配用例 | 实际模型词表上的候选屏蔽与生成终止 |
| `test-json-schema-to-grammar` | 既有 Schema 转换用例与规则预期 | 本次 Schema 的完整约束和 20 次生成 |
| `test-jinja` / `test-chat-template` | Jinja 与已有模板用例 | 当前 GGUF 模板、BOS、thinking 前缀和输入字节 |

### 第 6 周验收清单

- [ ] 能从 GGUF 元数据追到当前 tokenizer, 解释六类 round-trip 输入的差异。
- [ ] 能区分 token piece、词表内部文本、UTF-8 字符与 EOG。
- [ ] 能讲清 sampler 所有权、apply/accept/reset/clone, 不重复或遗漏 accept。
- [ ] 能用数值解释 temperature 与 top-p 顺序导致的候选差异。
- [ ] 能解释 grammar 的前缀状态、`-INFINITY` 屏蔽、UTF-8 与 EOG 条件。
- [ ] 能区分 common 的 grammar 检查/重采样与自建 grammar-first 链。
- [ ] 观测 sampler 的固定 logits、entropy、clone/reset 和透明性测试通过。
- [ ] Planner 正式矩阵连续 20 次完整结束, JSON 解析和 Schema 校验均通过, 原始失败记录保留。
- [ ] 能解释模板格式化、语法约束、Schema 校验、业务判断的不同职责。
- [ ] 能在 10 分钟内不看源码讲清本周生成控制链, 并指出全部状态的所有者。

## 本周完成包

```text
/home/king/llama-study/
  experiments/
    week6-sampler-study.cpp
  notes/
    week6-day1.md ... week6-day7.md
    week6-review.md
  results/week6/
    commit.txt
    tokenize-help.txt
    completion-help.txt
    tokenizer-roundtrip.csv
    sampler-stages.csv
    decision.gbnf
    decision-valid.txt / decision-prefix.txt / decision-invalid.txt
    grammar-validation.txt
    planner.schema.json
    planner.gbnf
    planner-prompt.txt
    schema-conversion.stderr.txt
    observer-tests.txt
    observer-trace.csv
    planner-cli.stdout.txt / planner-cli.stderr.txt
    planner-runs/
      run-01 ... run-20/
        config.json
        prompt.txt
        output.json
        tokens.txt
        validation.json
        trace.log
    failures/
    planner-summary.csv
    ctest.txt
```

日记保留事实与未验证假设, 周报告整理调用链、失败原因和适用边界。交给第 8 周的核心资产是 Planner Schema、输入模板约定、完整输出校验、请求间状态隔离和 observer; 第 8 周再增加角色调度与 Reviewer 闭环。
