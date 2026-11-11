# week8 day5: 单次模型加载、三角色骨架与首轮闭环

本文回答[第 8 周学习路线](../../008-llama-cpp-week8-daily-learning-route.md)中 Day 5 的当天输出与验收。源码基准为 `697b4bbf4633a38abfd66f7cc54a167095aedef3`。

本文给出与 [Day4 设计](week8-day4.md)一致的最小工程职责、初始化/生成流程、角色 prompt 和 smoke 验收记录格式。本次交付为答案文档, 未另建仓外工程或可执行的 `llama-multi-agent-demo`; 按要求暂时跳过实际构建和推理实验, 没有生成 `results/week8/demo-smoke/` 运行数据。因此“可构建并运行”这一项仍待完成, 下文示例不是实测结果。

## 1. 三角色骨架与构建边界

```text
application
  -> model_runtime: backend + one model + templates
  -> planner_session: history P + context P + sampler P
  -> worker_session: history W + context W + sampler W
  -> reviewer_session: history R + context R + sampler R
  -> generation_engine: format / tokenize / prefill / sample / decode
  -> protocol: parse / schema checks / application checks
  -> orchestrator: P -> W -> R -> accepted / revision_required / error
  -> trace_sink: immutable inputs and observed results
```

首日骨架只要求串通 P/W/R 一次。合法 revise 应保留候选和反馈并返回 `revision_required`, 不能写成 accepted; Day6 再接上第二轮并执行 Day4 的完整两轮状态机。真实模型不必首轮 accept, 但三个角色的输出必须能被正确分类。

未来最小工程可组织如下, 路径是计划, 不是已生成的文件:

```text
/home/king/llama-study/multi-agent/
  CMakeLists.txt
  README.md
  design.md
  src/main.cpp
  src/runtime.h / runtime.cpp
  src/protocol.h / protocol.cpp
  src/orchestrator.h / orchestrator.cpp
  schemas/planner.schema.json
  schemas/reviewer.schema.json
  tests/cases/
```

构建时使用当前 checkout 的 CMake targets `llama` 和 `llama-common`, 通过源码子目录或明确配套的构建产物接入, 不混用系统其他版本的头文件/库。公共模板/采样来自 common, 不需要链接 CLI、Server 或实现新的 HTTP route。参考 [common CMake](../../../../common/CMakeLists.txt) 与 [llama-simple CMake](../../../../examples/simple/CMakeLists.txt)。

仓外工程可以关闭上游 examples/tools/server/UI 的构建以减少无关工作, 但要保留 common 和所选 backend。它与本周用于研究 CLI/Server 的 `build-study-debug/release` 是不同构建用途。本文不把待实现的 `--model/--task/--task-file/--output-dir` 参数写成已有工具接口。

## 2. 一次加载与 RAII 的具体选择

第一版选择 `common_init_from_params(params, true)` 的 model-only 路径。初始化顺序:

1. 建立所需 backend 生命周期, 准备 CPU、本地模型、线程、context 和采样配置。
2. runtime 调用一次 `common_init_from_params(params, true)`, 持有返回的 `common_init_result_ptr`, 检查其 `model()` 非空。
3. 从这个长期存活的 owner 借用同一 `llama_model *`, 调用 `common_chat_templates_init()` 建立模板资源。
4. 三次调用 `llama_init_from_model(shared_model, cparams)`, 各自 `n_ctx=2048`、`n_seq_max=1`, 检查返回值和实际容量。
5. 为每个 session 建立独立 history; sampler 在每次生成开始时按角色重新创建, Planner/Reviewer 带各自约束。

model-only 路径在加载 model 后提前返回, 不创建正常完整初始化中的默认 context/samplers, 因而不会无意多保留第四个 context。它也跳过后续完整初始化附带的工作, 不能假设正常路径中的 warmup、sampler 配置等已经替三个角色完成。

[common.h](../../../../common/common.h) 与 [common.cpp](../../../../common/common.cpp)明确 `common_init_result` 的所有权。session 的 context 使用 `llama_context_ptr` 等已有 RAII 类型, sampler 使用 [sampling.h](../../../../common/sampling.h) 的 `common_sampler_ptr`。发生部分初始化失败时, 已创建对象自动释放, 不开始角色生成链。

### 如何证明只加载一次

后续 trace 应在真正的 runtime 加载入口记录 `model_load_attempts/model_load_successes` 和应用分配的 model ID。成功 smoke 的目标为加载尝试 1、成功 1, 三个 session 的 `llama_get_model(context)` 与 runtime model 相同, 同时 context 对象各不相同。

只写一个计数器常量为 1, 或看到模型文件只存在一份, 都不是加载次数证明。还要检查调用路径: session 初始化只调用 `llama_init_from_model()`, 不调用完整模型加载函数。当前仅完成这个源码/设计解释, 没有实测指针或加载日志。

## 3. 三角色输入与 prompt

固定 smoke 任务为: `为理解 KV Cache 制定三项练习。每项包含一个明确动作和一个可检查产出。`

| 角色 | system prompt 的要求 | user/history 数据 |
| --- | --- | --- |
| Planner | 只给符合 Planner Schema 的计划; goal 简短; 1-3 步, 至少一个 worker; 不增加预算或调用外部工具 | 原任务、字段约束与规划要求 |
| Worker | 根据已校验计划写三项练习; 每项有动作和产出; 返回候选正文 | 原任务、计划中的 worker 步骤和必要审查要求 |
| Reviewer | 对照原任务检查候选; 只给 decision/feedback; revise 必须指出具体修改点 | 原任务、合法计划、当前 Worker 候选、固定检查项 |

Reviewer 的固定检查项是: 恰有三项练习、每项有明确动作与可检查产出、没有把 KV 当作原始聊天记录等明显概念错误。这些要求写入输入, 但模型声称 accept 后仍需人工检查内容。

使用 `common_chat_templates_inputs.messages` 表达 system/user/assistant, 经 `common_chat_templates_apply()` 得到 prompt。Planner/Worker/Reviewer 是应用身份, 不是 chat template 的新 role。不得把其他角色的整个 private history 自动拼到当前角色消息中。

第二轮 Worker 的输入沿用 Day4 契约: 原任务 + 已验证计划 + W1 + 合法 feedback。Reviewer 第二轮只审查 W2。角色信息的传递应能在实际 messages 和 token 记录中解释, 不用模型是否复述某个标记来替代隔离检查。

## 4. 每次生成的 API 流程

| 步骤 | 使用的接口/动作 | 必须维持的条件 |
| --- | --- | --- |
| 重置本角色 | `llama_get_memory()`、`llama_memory_clear()`; 重建 common sampler | 清理上一调用的 KV 与采样状态, 保留本次需要的文本 history |
| 模板化 | `common_chat_templates_apply()` | 完整消息重新格式化, 保存模板配置和实际 prompt |
| 分词 | `common_tokenize()` 等已有辅助 | special token/BOS 处理与实际模板一致, 不重复添加 BOS |
| 容量检查 | 实际 prompt tokens + 生成预留 | 超限在 prefill 前返回 context_limit |
| Prefill | `llama_batch_init()`、`common_batch_add()`、`llama_decode()` | seq=0, position 从 0 递增, 按逻辑 batch 容量分块 |
| 首次采样 | `common_sampler_sample(sampler, ctx, idx)` | 读取完整 prompt 最后 token 对应的有效 logits |
| 接受 token | `common_sampler_accept(sampler, token, true)` | 每个生成 token 恰好一次, 不漏掉 grammar 状态更新 |
| 后续生成 | 非 EOG token 加入下一次 batch, 再 decode/sample | 递增 position, 检查每次 decode 返回值与预算 |
| 结束 | EOG/预算/超时/取消/错误 | 返回原始文本、原因和计数, 不自行进入下一角色 |
| 校验提交 | protocol -> orchestrator | 只有成功生成且验证通过才写入本阶段业务结果 |

prompt token 如需进入 repetition penalty/history, 使用 `common_sampler_accept(..., false)`, 不把整个 prompt 当生成正文推进输出 grammar。先创建全新的 sampler, 也避免上次 grammar 停在末尾、旧 RNG/penalty 状态残留。

`llama_memory_clear(mem, false)` 清除 memory metadata, 不保证数据缓冲被清零; `true` 同时清理数据缓冲。Day4 所需的是旧序列不再参与新生成, 应明确所选清理口径, 不能把 metadata 清理说成所有内存内容已擦除。第一版可使用 `true` 简化清理含义, 并把额外开销计入时间。

Prefill 的最后一个 prompt token 必须请求输出 logits。独立单 seq、当前仅一个所需输出时可以明确使用该输出或 `-1`; 若以后做多 seq, 必须像 Server 那样维护下标映射, 不能把这条简化扩大到所有 batch。

核心 `llama_sampler_sample()` 已经 accept 所选 token; common 的 `common_sampler_sample()` 则要显式配套 accept。本设计统一使用 common 两步接口, 不混用导致重复 accept。见 [llama.h](../../../../include/llama.h)、[sampling.cpp](../../../../common/sampling.cpp) 和 [parallel.cpp](../../../../examples/parallel/parallel.cpp)。

停止后最后的 EOG 不需要再次 decode, 达上限的最后 token 也可能没有送入后续 decode。因此 `generated_tokens`、`decoded_generated_tokens` 和输出字符数必须分开。decode 非零时不从可能失效的 logits 继续采样, 按 Day4 的错误语义返回并清理部分状态。

## 5. Planner/Reviewer grammar 与完整验证

分别使用 Day4 的两个 Schema, 不能共用一份已经推进到某个位置的 grammar sampler。`json_schema_to_grammar()` 与模板接口可提供对应约束, 但模板结果不仅有 prompt, 还可能携带 `grammar/grammar_lazy/grammar_triggers/preserved_tokens/additional_stops/parser` 等信息。

如果采用模板生成的结构化输出路径, 必须将相关 grammar 和解析/停止配置按其契约接入, 不能只拿 prompt、丢掉触发条件, 再假设从第一个生成 token 就套 JSON grammar 一定正确。若选择纯 JSON 正文模式, 则需先确认模板前缀与 reasoning 设置支持该模式。

本地 DeepSeek-R1 蒸馏模型可能带 reasoning。`enable_thinking=false` 是接口输入, 不保证所有模型模板都按此关闭思考。当前尚无本次真实运行证据, 应先以第 6 周方法确认实际模板、输出起点和 grammar 生效区域, 再做 smoke。不能用“找到某个花括号”或“补括号”把截断/混杂文本伪装成合法协议。

成功顺序是: 保存 raw output -> 检查 stop reason -> 取已定义边界内的完整协议正文 -> 严格 JSON parse -> Schema -> 应用规则 -> 写入 plan/review。JSON 解析成功不代替 Schema, Schema 成功不代替候选内容正确。

## 6. 首轮 smoke 记录契约

以下是待实现的 `results/week8/demo-smoke/` 内容, 本次均未产生:

```text
demo-smoke/
  config.json
  task.txt
  runtime.json
  planner.messages.json / planner.prompt.txt / planner.tokens.json
  planner.raw.txt / planner.validation.json
  worker-r1.messages.json / worker-r1.prompt.txt / worker-r1.tokens.json
  worker-r1.raw.txt / worker-r1.validation.json
  reviewer-r1.messages.json / reviewer-r1.prompt.txt / reviewer-r1.tokens.json
  reviewer-r1.raw.txt / reviewer-r1.validation.json
  trace.jsonl
  final.json
```

每次调用至少记录:

```text
run_id, role, round, model_id, context_id, seq_id,
input_tokens, generated_tokens, decoded_generated_tokens,
seed, sampler_config, grammar_id,
format_ms, tokenize_ms, prefill_ms, decode_ms, sample_ms, total_ms,
stop_reason, protocol_valid, validation_error, task_status
```

Planner 不属于 Worker/Reviewer round, 可记录 `round=0`; Worker/Reviewer 首轮记录 1。三个 context 都用 seq 0, 但 context ID 分别为 P/W/R 对应的真实对象标识。业务稳定 ID 与原始指针地址应分开, 跨进程不能靠地址相同判断同一个模型。

| 验收项 | 后续必须看到的证据 | 本次状态 |
| --- | --- | --- |
| 工程可构建 | 配套 checkout、完整配置和构建日志、可执行文件 | 未搭建/未编译 |
| 只加载一次 | runtime 的加载日志/计数, 三 context 共享 model 的检查 | 未运行 |
| Planner 完整合法 | raw output、EOG/明确停止原因、完整解析及应用验证 | 未运行 |
| Worker 使用计划 | 实际输入含合法 plan, 候选非空且有三项练习 | 未运行 |
| Reviewer 合法 | decision 为 accept/revise, revise feedback 有效 | 未运行 |
| Trace 完整 | 三次调用输入/输出、context/seq、原因和计时 | 未采集 |
| 首轮结果准确 | accept 标为模型接受; revise 标为待修订; 错误不继续 | 已给出契约, 待运行验证 |

JSON 被 token 上限截断时, 应如实返回 token_limit, 再在预算内调整 prompt 或生成上限后重新实验。调整需另记配置, 不能修补原始输出后将原实验标为成功。

## 7. 每个 owner 与错误后的控制权

session 持有本角色 history/context/sampler; runtime 持有共享 model; generation_engine 持有本次 batch 和输出; orchestrator 持有已经验证的 plan/candidate/review 和任务状态。

generation_engine 报错后返回明确结果, protocol 报错后返回解析/校验原因。真正决定“不调用下一角色”的是 orchestrator: Planner 失败就没有 Worker, Worker 失败就没有 Reviewer, Reviewer 非法就没有下一轮。trace_sink 记录已经发生的失败, 不发起重试。RAII 和 session 重置负责释放/清理资源, 不能把停止控制寄托在析构函数里。

本次完成了骨架设计、单次加载解释、角色 prompt、生成/验证流程和完整 smoke 证据清单。实际工程尚未落地; 加载次数实测、三角色推理、计时以及人工内容验收按要求暂时跳过, Day5 的运行验收尚未通过。
