# week6 day5: Chat template、Planner Schema 与应用协议

本文回答[第 6 周学习路线](../../006-llama-cpp-week6-daily-learning-route.md)中 Day 5 的当天输出与验收。

核对基准为当前 checkout `a8741cc99dadd9bea67ad91c225b69ce772909f6`。已阅读源码并只读检查本地 GGUF metadata; 未调用 C++ 模板渲染、Python/C++ Schema 转换器、模型生成或验证器。按要求跳过手动实验, 将 Schema、GBNF 和 prompt 内容草案集中写在本文; 不把静态推导标成运行日志。

## 1. 输入格式化与输出约束的两条路径

```text
GGUF tokenizer.chat_template / 显式模板覆盖
  + messages + 模型 BOS/EOS + 模板参数
  -> common_chat_templates_init()
  -> common_chat_templates_apply(use_jinja=true)
       -> Jinja lexer -> parser/program -> runtime
       -> common_chat_params: prompt + generation_prompt + parser 等
  -> 按特殊 token/BOS 约定 tokenize
  -> prompt token IDs -> prefill/decode -> logits

Planner JSON Schema
  -> json_schema_to_grammar(schema, true)
  -> 非 lazy grammar sampler
  -> 对 logits 的允许 token 集合施加约束
  -> sample + accept -> 生成 token pieces
  -> 完整 JSON 解析 -> 独立 Schema 校验 -> 应用业务检查
```

模板属于应用输入/输出协议层, 模型计算图处理后续 token/embedding, 不直接解释 Jinja 或 JSON Schema。

`common_chat_templates_inputs` 承载 messages、grammar、json_schema、add_generation_prompt、use_jinja、enable_thinking、tools 等输入。`common_chat_params` 除 prompt 之外还携带 generation_prompt、grammar、grammar_lazy、parser、thinking 标签和停止标记等, 不能只保留 prompt 就假定已经保留完整协议。

核心 `llama_chat_apply_template()` 是预定义模板的 C++ 实现, API 注释明确它不使用 Jinja parser。通用 Jinja 运行应使用 common 层; 当前仓库的独立引擎在 `common/jinja`, 不应称为 Minja。

## 2. 当前模型的模板与生成前缀

模型为 `/home/king/llama.cpp/deepseek-r1-1.5b-f16.gguf`。从该文件实际读取的架构为 `qwen2`, BOS 为 151646, EOS 为 151643, `add_bos=true, add_eos=false`。此处的模板信息来自 GGUF, 不是根据模型名称套用仓库中另一份模板。

### 模板的静态行为

| 对象/条件 | GGUF 内模板源码所表达的行为 |
| --- | --- |
| 模板开头 | 输出 `bos_token`, 然后输出收集到的 system 内容 |
| user 消息 | 输出 `<｜User｜>` 后跟内容 |
| 普通 assistant 消息 | 输出 `<｜Assistant｜>`、内容、`<｜end▁of▁sentence｜>` |
| 历史 assistant 内容含 `</think>` | 普通消息分支仅保留最后一个 `</think>` 之后的内容 |
| `add_generation_prompt=true`, 非 tool 结尾 | 追加 `<｜Assistant｜><think>` 和一个换行 |
| `enable_thinking` | 该 GGUF 模板源码没有引用这个变量 |

模板末尾字符串字面量中的 `<think>` 后是字节 `5C 6E`, 即 Jinja 源码转义 `\n`; 当前 lexer 会把它解析成 LF `0A`。这是对模板源码与 lexer 的静态推断, 实际 `common_chat_params.prompt` 字节仍待渲染核对。

`common_chat_templates_init()` 优先使用显式 override; 没有 override 时读取模型模板。当前模型已有模板, 无需回退到默认 ChatML。当前专用模板分支的匹配条件不覆盖这份 R1 模板; 在默认 `force_pure_content=false` 的 Jinja 路径中, 源码会继续尝试 differential autoparser, 其实际结果及是否成功尚未运行确认。

### BOS 为什么要结合 common 后处理判断

当前 `common_chat_template_direct_apply_impl()` 在模型配置 `add_bos=true` 且渲染结果以 BOS 文本开头时, 会去掉开头的 BOS 文本。这是为后续 tokenize 自动添加 BOS 留出空间。因而要区分:

```text
原始模板渲染: BOS文本 + 正文
common direct-apply 后处理: 正文
tokenize(add_special=true, parse_special=true): BOS_ID + 正文IDs
```

本周草案沿用这一约定。若改为直接使用未经 common 后处理的原始渲染文本, 必须重新核对 add_special, 否则会产生重复 BOS。模板只应用一次, 不再把已格式化 prompt 当 user 内容包进另一层模板。common 返回的 generation_prompt 用于描述/解析末尾前缀, 也不能无条件再次追加到已经包含它的 prompt。

### 同一组消息的 false/true 对照草案

固定消息为 `user: Review a C++ function.`、`assistant: Send the function.`、`user: Return a short plan.`。无 tools, 不续写最后一条历史 assistant, 不额外添加 system。按 GGUF 模板和 direct-apply 的 BOS 后处理推导:

`add_generation_prompt=false`:

```text
<｜User｜>Review a C++ function.<｜Assistant｜>Send the function.<｜end▁of▁sentence｜><｜User｜>Return a short plan.
```

`add_generation_prompt=true` 在上述末尾追加:

```text
<｜Assistant｜><think>
```

这里追加片段的结尾有一个 LF; false 版本自身不额外添加 LF。以上只是 direct-render 分支的预期, 未取得默认 autoparser 路径的真实 prompt 文件或 token IDs。

## 3. Planner 的最小 Schema 草稿

以下是 `planner.schema.json` 的完整内容, 与路线约定一致。`goal` 为 1..48 个字符, `steps` 为 1..3 项, 每项只能含 role/task, task 为 1..64 个字符。

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

类型、required 和 additionalProperties 解决不同问题: properties 本身不意味着字段必须出现; required 不限制额外字段; enum 不判断任务是否合理。字符串长度按解析后的字符计数, 不能按 UTF-8 字节或 JSON 源文本中转义序列的长度计数。

## 4. GBNF 规则草案与转换证据边界

C++ 明确输出 GBNF 的入口是 `json_schema_to_grammar(schema, true)`。第二个参数在开启 llguidance 的构建中尤其重要, 默认 false 可能返回 `%llguidance` 格式。普通 CLI 的 `--json-schema-file` 将文件转换后放入 `params.sampling.grammar`, 不会自动把字段语义插进 prompt。特殊 chat/tool 分支可能额外处理 Schema, 需另看分支。

当前 C++ 转换器存在 object required/additionalProperties、enum、array minItems/maxItems 和 string minLength/maxLength 的处理。为便于审阅, 下面给出手写的 `planner.gbnf` 结构草案, **不是执行转换器获得的文件**, 也未调用 parser 验证:

```gbnf
root ::= "{" ws "\"goal\"" ws ":" ws goal ws "," ws "\"steps\"" ws ":" ws steps ws "}" ws
steps ::= "[" ws step (ws "," ws step){0,2} ws "]"
step ::= "{" ws "\"role\"" ws ":" ws role ws "," ws "\"task\"" ws ":" ws task ws "}"
role ::= "\"worker\"" | "\"reviewer\""
goal ::= "\"" char{1,48} "\""
task ::= "\"" char{1,64} "\""
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n\r]*
```

这份草案固定对象字段顺序为 goal/steps 与 role/task, 有意采用 Schema 合法表示的一个子集; JSON Schema 本身不规定字段顺序。char 参考转换器的基础规则, 对部分合法 JSON 表示也有限制, 不能宣称它与所有 JSON Schema 语义等价。

即使生成规则含 `{1,48}`, 也要独立校验解析后的字符长度: 例如代理对形式的两个 `\uXXXX` 转义在此规则中匹配两次 char, 在符合 Unicode 语义的解析器中可能合成一个非 BMP 字符。规则转换成功只能说明转换过程完成, 不证明全部关键字、Unicode 或组合语义完整实现。

后续实际转换需保存 Python stdout 为 `planner.gbnf`, stderr 为 `schema-conversion.stderr.txt`, 并检查退出状态、警告及所有引用规则。C++ 转换结果另行核对, 不能由 Python 一方通过替代另一方。此次按要求不执行这些手动步骤, 不报告“无警告”或“转换通过”。

## 5. Planner prompt 草案与纯 JSON 输出边界

本周的协议是: 从第一枚新生成 token 起使用非 lazy JSON grammar, prompt 本身不进入输出 grammar, 输出文件仅保存新生成的非 EOG pieces。

为隔离 reasoning/tool 解析, 后续最小程序可显式选用 `use_jinja=true, force_pure_content=true, add_generation_prompt=true`, 使用 GGUF 模板格式化, 然后独立安装上面的输出 grammar。当前 pure-content 分支直接渲染并返回内容 parser, 不会自动把传入 Schema 变成所需的独立 grammar; 约束必须在 sampler 初始化处落实。这是本文建议的待验证实验配置, 不是已运行的配置。

当前 common 初始化另有 generation_prompt 预填充逻辑: 对非 lazy 的 OUTPUT_FORMAT/TOOL_CALLS grammar 会预先 accept 该前缀。不要把这里的纯 JSON 根规则与带 thinking 标记的生成前缀直接拼成这种配置, 否则前缀不属于 JSON grammar。本草案用独立 core grammar 从根规则开始; 若改用 common, 可将转换后的 GBNF 作为 USER grammar, 明确不预填充模板前缀, 并再核对初始化与接受路径。

这也不会自动删掉模板的 `<think>` 前缀。该 GGUF 模板没有 `enable_thinking` 判断, 因此仅设 false 不能据此断言前缀会消失。本周保留其为输入上下文, 新生成区域仍强制纯 JSON, 不另外输出思考文本或 `</think>`。这可能使模型处于不自然的续写条件、影响计划质量, 影响大小需实测, 不能把格式正确视为质量保证。按 reasoning 结束再启动 lazy grammar 不在本次范围。

### 消息内容与格式化文本

system 消息草案:

```text
You are a planner. Return only one JSON object with goal and steps. goal must have 1 to 48 characters. steps must have 1 to 3 objects, each containing only role and task. role must be worker or reviewer. task must have 1 to 64 characters. Use concrete steps relevant to the user task. Do not output markdown, explanations, or thinking text.
```

user 消息草案:

```text
Create a short plan to review a C++ function for correctness and edge cases.
```

`planner-prompt.txt` 的预期内容如下, 基于 direct-apply 移除 BOS 后的约定。最后恰有一个 LF, 中间不人为插入空行:

```text
You are a planner. Return only one JSON object with goal and steps. goal must have 1 to 48 characters. steps must have 1 to 3 objects, each containing only role and task. role must be worker or reviewer. task must have 1 to 64 characters. Use concrete steps relevant to the user task. Do not output markdown, explanations, or thinking text.<｜User｜>Create a short plan to review a C++ function for correctness and edge cases.<｜Assistant｜><think>
```

这是根据已读模板推导的内容草稿, 不是模型格式化函数的实测输出文件。后续应保存实际 prompt 字节及 hex, 比较末尾 LF、BOS 数量、generation_prompt 和实际 token IDs。

### 输入来源标记

Jinja 的 `jinja::string` 可以保留各片段的 `is_input` 来源标记, `global_from_json(..., mark_input=true)` 开启输入标记。模板产生的角色分隔符和用户输入中的同名字面文本不应被无条件当作同等来源的控制 token。

不过当前 direct-apply 最终调用 `parts->as_string().str()` 返回普通字符串; 保存一个 prompt.txt 或把它整体交给 `parse_special=true` 不会自动保留逐片段来源保护。本文固定的消息不含用户伪造的特殊标记; 若推广到任意用户输入, 需要另外明确保留来源标记及特殊 token 的处理方式, 不能宣称启用 Jinja 就已经完成全部防护。

## 6. 三层验证与正反例

| 层次 | 必须检查 | 不能保证 |
| --- | --- | --- |
| JSON parser | 完整消费全部输出, 除合法空白外无尾随文本 | 不保证 goal/steps 存在或类型正确 |
| 独立 Schema validator | 字段、类型、required、enum、长度、数组上下限及额外属性 | 不保证步骤与任务相关、可执行 |
| 应用业务检查 | 与输入目标相关, 行动明确, 顺序合理, 满足额外角色规则 | 不能由语法层替代, 也不表示步骤已经被实际执行 |

通过三层的示例草案:

```json
{
  "goal": "Review C++ function",
  "steps": [
    {"role": "worker", "task": "Check logic, bounds, and error handling"},
    {"role": "reviewer", "task": "Verify findings and suggest edge case tests"}
  ]
}
```

以下判断按定义静态给出, 未运行 validator:

| 输入/变体 | JSON 层 | Schema 层 | 业务层 |
| --- | --- | --- | --- |
| `{"goal":"Review","steps":"check code"}` | 可解析 | steps 不是数组, 失败 | 无需进入 |
| `{"goal":"Review","steps":[]}` | 可解析 | minItems 失败 | 无需进入 |
| role 改为 `planner` | 可解析 | enum 失败 | 无需进入 |
| 删除 goal 或新增顶层 debug | 可解析 | required 或 additionalProperties 失败 | 无需进入 |
| task 为空或 65 个普通 ASCII 字符 | 可解析 | minLength/maxLength 失败 | 无需进入 |
| JSON 后附 `Done` 或包 Markdown 代码围栏 | 完整消费失败 | 无需进入 | 无需进入 |
| 截断在字符串/数组中间 | 失败, 可能只是合法前缀 | 无需进入 | 无需进入 |
| goal 为 `Water plants`, task 为 `Water the garden` | 可解析 | 可通过 | 与 C++ 审查无关, 失败 |

此 Schema 允许只有 reviewer 的计划, 也不禁止两个 worker。“必须先 worker 后 reviewer”若成为需求, 需加独立业务规则并明确验收, 不能声称目前 Schema 已经保证。grammar 对格式的约束也不保证模型断言真实或任务实际完成。

## 7. 交付物与验收状态

路线中的结果目录是 `/home/king/llama-study/results/week6/`。按本次“答案写入文档、手动实验暂时跳过”的要求, 本文保留可审阅草案, 未另外生成下列实验结果文件。

| 计划文件 | 本文已提供 | 尚未取得的证据 |
| --- | --- | --- |
| `planner.schema.json` | 完整 JSON 内容草稿 | 独立文件及实际 validator 验证 |
| `planner.gbnf` | 规则结构草案、关键词对应和语义边界 | Python/C++ 实际转换输出及 parser 验证 |
| `planner-prompt.txt` | 消息内容、格式化预期和参数约定 | C++ 渲染后的真实字节/token IDs |
| `schema-conversion.stderr.txt` | 需检查的日志项目 | 尚未执行转换, 不存在可报告的成功/失败日志 |

- [x] 已给出 prompt/Schema 两条路径、当前 GGUF 模板、生成前缀和 BOS 约定。
- [x] 已给出完整 Schema、GBNF 与 prompt 草案, 区分三层校验并列出正反例。
- [x] 已说明 reasoning 前缀、输入来源标记及固定纯 JSON 协议的边界。
- [ ] 模板 false/true 实际对照、Schema 转换日志、规则验证及模型生成: 按要求暂时跳过。
- [ ] 第 7 天连续 20 次生成验收: 不属于本次 Day1-Day5 的实测结论, 未执行。

源码入口: [chat.h](../../../../common/chat.h)、[chat.cpp](../../../../common/chat.cpp) 的 init/apply/direct-apply/generation-prompt 及 specialized-template 分派、[Jinja README](../../../../common/jinja/README.md)、[lexer.cpp](../../../../common/jinja/lexer.cpp)、[json-schema-to-grammar.h](../../../../common/json-schema-to-grammar.h)、[json-schema-to-grammar.cpp](../../../../common/json-schema-to-grammar.cpp)、[Python 转换器](../../../../examples/json_schema_to_grammar.py)、[arg.cpp](../../../../common/arg.cpp)、[test-chat-template.cpp](../../../../tests/test-chat-template.cpp)、[test-json-schema-to-grammar.cpp](../../../../tests/test-json-schema-to-grammar.cpp)。
