# week6 day2: Sampler 生命周期、API 契约与状态所有权

本文回答[第 6 周学习路线](../../006-llama-cpp-week6-daily-learning-route.md)中 Day 2 的当天输出与验收。

核对基准为当前 checkout `a8741cc99dadd9bea67ad91c225b69ce772909f6`。范围为 CPU、host sampling、普通自回归生成, 不涉及 backend sampling、speculative decoding、Mirostat、DRY 或 lazy grammar。本文为源码分析; 模型运行、断点和 checkpoint 实验按要求暂时跳过。

## 1. 生命周期与所有权图

```text
init: 创建 iface + ctx
  |
  +--> 每一步重建 candidates
         -> apply: 改 logits / 有效视图 / selected, 也可能推进 RNG
         -> 检查 selected 在 [0, size) 中
         -> id = data[selected].id
         -> accept(id): 提交一次历史/grammar 状态
         -> 应用按需 decode(id), 进入下一步
  |
  +--> reset: 各实现恢复自己的初始状态
  +--> clone: 复制当前可续写状态, 两个实例随后独立演进
  +--> free: 释放私有 ctx, 再释放外层 sampler
```

上图的显式 accept 适用于直接 `llama_sampler_apply()` 路径; 使用 `llama_sampler_sample()` 的 CPU 路径时, accept 已在内部完成。

```text
应用
  +-- owns llama_sampler chain
  |     +-- owns penalties: ring buffer + token counts
  |     +-- owns grammar: rules + stacks + partial_utf8 (若自行放入 chain)
  |     +-- owns top-k / top-p / temperature: 参数及必要缓冲
  |     +-- owns dist: seed + 当前 RNG
  |
  +-- 或 owns common_sampler
        +-- owns grmr: 独立 grammar sampler
        +-- owns rbudget: 可选 reasoning budget, 本周不启用
        +-- owns chain: 概率策略和最终选择器
        +-- owns prev: common 历史缓冲
        +-- owns cur: 候选存储; cur_p 是当前视图

model/vocab: grammar 引用词表, 不拥有词表, 必须活得足够久
iface: 公共 init 保存指针, 自定义接口表必须覆盖 sampler 的整个生命周期
```

### 所有权规则

| 操作 | 操作后的所有者 | 使用约束 |
| --- | --- | --- |
| `llama_sampler_init_*()` | 调用者 | 后续必须释放或转交 chain |
| `llama_sampler_chain_add(chain, child)` | chain | 调用者不能再单独 free child; 同一 child 不能加入两个 chain |
| `llama_sampler_chain_get()` | 仍是 chain | 返回借用指针, 没有转移所有权 |
| `llama_sampler_chain_remove()` | 接收返回指针的调用者 | 调用者负责 free 或交给另一 chain |
| `llama_sampler_clone()` | 新实例由调用者拥有 | 子 sampler 必须提供正确 clone 语义 |
| `llama_sampler_free(chain)` | 释放整个 chain | 递归 free 子 sampler, 然后释放 chain 的 ctx 和外层对象 |

自定义 sampler 的 free callback 负责释放自身 ctx; 外层 `llama_sampler` 由公共 `llama_sampler_free()` 的 `delete smpl` 释放。回调里再次删除外层对象会导致重复释放。

## 2. 两层 API 契约表

| 路径 | 如何选出 ID | 是否自动 accept | 正确调用约定 |
| --- | --- | --- | --- |
| `llama_sampler_apply(s, &cur_p)` | 顺序执行策略, 最终 greedy/dist 等填写 `selected` | 否 | 验证下标后取 `data[selected].id`, 显式 `llama_sampler_accept(s,id)` 一次 |
| `llama_sampler_sample(s,ctx,idx)` 的 CPU 路径 | 获取输出 logits, 构造候选, apply, 读 selected | 是, 内部调用 core accept | 返回后不要再 core accept |
| `common_sampler_sample(g,ctx,idx,grammar_first)` | chain 选择, 结合 grammar 检查和必要重采样 | 否 | 返回后调用 `common_sampler_accept(g,id,true)` 一次 |
| common 接受 prompt | 没有采样 | 由调用者显式提交 | `common_sampler_accept(g,prompt_id,false)`, 更新 chain/prev, 不把普通 prompt 提交给输出 grammar |

`selected` 是当前有效候选视图的下标。例如 `data[0].id=151643` 且 `selected=0`, 选中 ID 是 151643, 不是 0。top-k/top-p 等操作后不能假设数组下标等于 token ID。

当前核心 CPU sample 的主要过程为:

```text
ctx 的第 idx 个请求输出
  -> logits[id]
  -> cur = [{id, logit, p=0}, ...]
  -> cur_p = {data=cur.data(), size=n_vocab, selected=-1, sorted=false}
  -> llama_sampler_apply(smpl, &cur_p)
  -> assert selected 有效
  -> token = cur_p.data[cur_p.selected].id
  -> llama_sampler_accept(smpl, token)
  -> return token
```

这里候选字段 `p=0` 只是初始化值, 不是当前归一化概率。`idx=-1` 指向本次可用输出中的最后一行, 不能理解成任意 KV cell 的预测结果。当前函数还有 backend 已选出 token 时直接返回的分支, 它不走上述 CPU accept, 因而结论不能推广到所有 backend 路径。

common 层的 grammar 与 chain 分开持有, `common_sampler_accept()` 在 `is_generated=true` 且 grammar 适用时推进 grammar, 然后更新 chain 和 `prev`。completion 工具的生成 token 与 prompt 处理循环分别传 true 和 false, 是这份契约的直接使用例。

这不等于 common 初始化时永远不会提前 accept: 当前 `common_sampler_init()` 会对非 lazy 的 OUTPUT_FORMAT/TOOL_CALLS grammar, 在存在 generation_prompt 时预先接受该前缀的 token; USER grammar 不做这种预填充。它处理的是模板已经提供的生成前缀, 不是整段 prompt。Day 5 的纯 JSON 实验使用独立 core grammar, 起始状态应留在 JSON 根规则。

## 3. apply、accept、reset、clone 的状态边界

| 操作 | penalties | grammar | dist |
| --- | --- | --- | --- |
| apply | 按已接受历史改 logits, 不追加当前 token | 检查候选是否能延续前缀, 屏蔽非法 logit | 从当前 logits 求概率并选择; 通常消耗 RNG |
| accept | 更新滑动窗口和 token 次数 | 推进 stacks 与跨 token UTF-8 状态 | 当前实现无 accept callback |
| reset | 清空窗口及次数 | 核心 grammar sampler 重新构建初始语法状态 | 按初始 seed 重置 RNG |
| clone | 复制窗口和次数 | 复制 rules/stacks/partial_utf8, 修正栈内指针 | 复制当前 RNG 状态 |

dist 的空候选/单候选快速路径不消耗随机数, 不能用“每次 apply 恰好一次 RNG”作通用假设。调试时额外 sample 一次会改变通常路径的 RNG, 若通过核心 sample 还会额外提交历史。

`reset` 不是所有组件共同遵守的“回到上一 checkpoint”。固定显式 seed 的 dist reset 是从头开始 RNG 序列; `LLAMA_DEFAULT_SEED` 哨兵还可能重新取得随机 seed。clone 才是复制当前 RNG 位置的手段。只有 ctx 为空时, 未提供 clone callback 的 sampler 才可由公共 API 创建无状态副本; ctx 非空且没有 clone callback 会终止, 不会自动深拷贝。

### 当前 common reset 的实际限制

当前 `common_sampler::reset()` 只有:

```text
prev.clear()
llama_sampler_reset(chain)
```

`common_sampler_reset()` 转调这个方法, **没有重置独立的 `grmr` 和 `rbudget`**。所以不能把 `common_sampler_reset()` 当作新请求的完整初始化。普通独立请求应 free 并重新 `common_sampler_init()`; 自建 core chain 若包含 grammar, core chain reset 会向该子项转发 reset, 这与 common 的所有权结构不同。

`common_sampler_clone()` 会 clone grmr/rbudget/chain 并复制 prev。当前实现还直接复制 `cur_p` 结构, 其中 data 是临时视图指针; 不应把 clone 后遗留的候选视图当成独立快照。正常下一次 sample 会通过 `set_logits()` 重新绑定它。

## 4. checkpoint 的四列补充图

```text
同一个逻辑 checkpoint 边界
+--------------------+----------------------+---------------------+----------------------+
| memory/context     | pending input        | sampler state       | application history  |
+--------------------+----------------------+---------------------+----------------------+
| 已 decode 的 KV    | 已选但尚未 decode   | RNG 当前位置        | 消息、输出字节       |
| pos / seq metadata | 的 token / pos / seq | penalties 窗口计数  | 已接受 token IDs     |
| 对应输出的有效性   | 下一次 batch 信息    | grammar 前缀状态    | 停止条件、计数、配置 |
+--------------------+----------------------+---------------------+----------------------+
```

假设刚完成 `llama_sampler_sample()` 并得到 token t, 但还没 decode t。sampler 已经 accept t, KV 却尚未包含 t, 应用应把 t 记为 pending。恢复后要 decode t, 不能再 accept t 或重新采样一个 token 替代它。若选在 decode 后保存, 则必须相应调整 pending 和输出行边界。

| 方法 | 能解决的问题 | 不能自动解决的问题 |
| --- | --- | --- |
| 保存/恢复 context state | 恢复接口覆盖的模型推理状态 | 外置 sampler RNG、grammar、应用输出和 pending 输入不会一起自动保存 |
| reset | 准备 sampler 的初始状态, 以具体实现为准 | 不能恢复中途 RNG; common reset 也不覆盖独立 grammar |
| clone | 在进程内复制当前 sampler 的可续写状态 | 不是磁盘序列化格式, 不保存模型 KV 和应用状态 |
| 重新 accept 历史 | 对本周组件重建 penalties, 按生成历史重建 grammar | dist 的 accept 不推进 RNG, 所以重放 token 历史不能恢复 RNG |
| 仅重新设置相同 seed | 重启同一个随机序列 | 不知道之前消耗多少随机数, 尤其 common 可能发生重采样 |

重放时必须区分 prompt 与生成 token: prompt 只进入需要其历史的 chain, grammar 从约定的输出边界开始接收生成 token。对于任意自定义 sampler, 还需检查其状态是否全部可由 accept 重建。

## 5. 三类错误调用及修正

| 错误序列 | 后果 | 修正 |
| --- | --- | --- |
| core `sample -> accept(id)` | 同一 token 被记两次, frequency 计数增加、窗口提前挤出旧 token; grammar 把 piece 消费两遍, 可进入错误状态 | core CPU sample 返回后直接进入应用处理/decode |
| common `sample -> decode`, 漏 common accept | 模型历史前进, penalties/grammar 仍停在旧前缀; 后续结果可重复或违反完整语法 | 在选中后、下一步采样前执行 common accept 一次 |
| 两请求共享一个可变 sampler, 或同一子对象交给两个 chain | 历史、grammar、RNG 相互污染; 并发有竞争, 双重所有权还会重复 free | 每请求独立初始化; 分叉时 clone, 同时复制匹配的 context/pending/app 状态 |

还应避免对包含输出 grammar 的 core chain 直接 accept 整段 prompt。可以把 grammar 单独持有, 或分别初始化概率历史和输出语法; common 层已有对应的 `is_generated` 区分。

## 6. 验收与跳过说明

- [x] 已给出生命周期、所有权图、两层 API 契约和 checkpoint 四列图。
- [x] 已解释重复 accept、漏 accept、共享 sampler 的后果及修正方式。
- [x] 已根据当前源码注明 common reset、clone 临时候选视图和 RNG 恢复的限制。
- [ ] GDB 观察调用次数、模型续写、clone/reset/checkpoint 对照: 按要求暂时跳过, 未宣称运行验证通过。

源码入口: [llama.h](../../../../include/llama.h) 的 `llama_sampler_i` 与 chain API、[llama-sampler.cpp](../../../../src/llama-sampler.cpp) 的公共 API/chain/dist/penalties/grammar 实现、[llama-sampler.h](../../../../src/llama-sampler.h)、[sampling.cpp](../../../../common/sampling.cpp) 的 `common_sampler` 和 sample/accept/reset/clone、[sampling.h](../../../../common/sampling.h)、[completion.cpp](../../../../tools/completion/completion.cpp) 的 prompt/生成 token 接受循环。
