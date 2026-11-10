# week6 day3: 固定 logits、采样顺序与概率观察

本文回答[第 6 周学习路线](../../006-llama-cpp-week6-daily-learning-route.md)中 Day 3 的当天输出与验收。

核对基准为当前 checkout `a8741cc99dadd9bea67ad91c225b69ce772909f6`, 仅讨论 host sampler。本文给出公式推演, 用 Python 标准库复核算术; 没有运行 llama.cpp sampler、模型或 GDB。实测 CSV 与断点实验按要求暂时跳过。

## 1. 策略对当前候选视图的影响

| 策略 | 主要动作 | `size/sorted/selected/p` 的边界 |
| --- | --- | --- |
| penalties | 用接受历史改 logit | 不截断 size; 启用时置 sorted=false; 不重新计算 p |
| top-k | 保留 logit 最大的 k 项 | 当前 k<=0 直接返回; 其他情形排序/截断有效视图, 不计算新 p |
| top-p | softmax, 按累计概率达到阈值且满足 min_keep 的位置截断 | 截断后的 p 来自截断前归一化, 保留部分的概率和可能小于 1 |
| temperature | T>0 时 `logit /= T` | 不重新计算 p, 不选择 token; 正有限温度下保持 logit 的大小关系 |
| greedy | 找最大 logit, 写 selected | 不需要 RNG, 不要求 p 已填好; 并列最大值取当前遍历中的第一项 |
| dist | 对有效视图重算概率并抽样 | 写 selected, 当前实现通常不排序; 多候选路径推进 RNG, 单候选直接 p=1 |

`data`、顺序和 `size` 共同决定有效视图, 不能只看原始分配的 vector 长度。本文四候选路径中没有 data 指针偏移, 不能推广为所有 sampler 都如此。

## 2. 固定输入

选择四个不同的教学 ID `[10,20,30,40]`, 与实际模型的语义无关。每条路径从同一份原始候选副本开始, 设 `selected=-1, sorted=false, p字段=0`。

| token ID | 初始数学概率 P | logit = ln(P) |
| --- | ---: | ---: |
| 10 | 0.55 | -0.597837001 |
| 20 | 0.25 | -1.386294361 |
| 30 | 0.15 | -1.897119985 |
| 40 | 0.05 | -2.995732274 |

“初始数学概率”由给定 logits 推导, 不代表候选结构的 `p` 字段初始化时就具有这些值。参数固定为 `top_p=0.85, min_keep=1, T=0.5`。

正温度变换满足:

```text
q_i = exp(logit_i / T) / sum_j exp(logit_j / T)
    = P_i^(1/T) / sum_j P_j^(1/T)
T=0.5 时权重为 P_i^2
```

## 3. 路径 A: top-p -> temperature -> dist

top-p 在原始分布上的累计概率为 `0.55, 0.80, 0.95, 1.00`。达到 0.85 时已经包含第三项, 所以保留 `[10,20,30]`, 而非只保留累计值小于阈值的前两项。

温度为 0.5 后, 三项权重为 `[0.3025,0.0625,0.0225]`, 总和 0.3875。dist 最终归一化分布为:

```text
ID:  [10,          20,          30]
q:   [0.780645161, 0.161290323, 0.058064516]
精确比例: [121/155, 25/155, 9/155]
```

| 阶段 | size / finite_logits | 当前 ID | 当前 logits | p 字段 / 概率和 | sorted | selected |
| --- | --- | --- | --- | --- | --- | --- |
| 原始副本 | 4 / 4 | 10,20,30,40 | ln(P) | 全 0, 尚未求概率 | false | -1 |
| top-p | 3 / 3 | 10,20,30 | -0.597837,-1.386294,-1.897120 | 0.55,0.25,0.15; 和 0.95 | true | -1 |
| temperature | 3 / 3 | 10,20,30 | -1.195674,-2.772589,-3.794240 | 仍是旧值, 已不代表当前 logits | true | -1 |
| dist | 3 / 3 | 10,20,30 | 同上一行 | 0.780645,0.161290,0.058065; 和 1 | true | 随机下标 0..2, 未实测 |

## 4. 路径 B: temperature -> top-p -> dist

先对四项缩放, 权重为 `[0.3025,0.0625,0.0225,0.0025]`, 总和 0.39。top-p 计算的分布为:

```text
[0.775641026, 0.160256410, 0.057692308, 0.006410256]
```

前两项累计 `0.935897436 >= 0.85`, 所以保留 `[10,20]`。dist 在这两项中重新归一化:

```text
ID:  [10,          20]
q:   [0.828767123, 0.171232877]
精确比例: [121/146, 25/146]
```

| 阶段 | size / finite_logits | 当前 ID | 当前 logits | p 字段 / 概率和 | sorted | selected |
| --- | --- | --- | --- | --- | --- | --- |
| 原始副本 | 4 / 4 | 10,20,30,40 | ln(P) | 全 0 | false | -1 |
| temperature | 4 / 4 | 10,20,30,40 | -1.195674,-2.772589,-3.794240,-5.991465 | 仍全 0, 尚未求概率 | false | -1 |
| top-p | 2 / 2 | 10,20 | -1.195674,-2.772589 | 0.775641,0.160256; 和 0.935897 | true | -1 |
| dist | 2 / 2 | 10,20 | 同上一行 | 0.828767,0.171233; 和 1 | true | 随机下标 0..1, 未实测 |

因此 ID 30 在 A 中概率约 5.806%, 在 B 中为 0。即使两次都抽到 ID 10, 也不能证明两个分布相同。top-p 的保留集合依赖概率质量, temperature 先改变质量再做 top-p, 与先截断再缩放不交换。

这里的具体数值是数学推演; C++ float 运算可有末位舍入差异。本例累计值离阈值足够远, 推演的保留集合没有临界舍入歧义。

## 5. 接受两次历史后的 penalties 推演

再次从原始四候选开始, 新建 penalties 状态, 设 `penalty_last_n=8, penalty_repeat=1.2, penalty_freq=0.4, penalty_present=0.1`, 显式 accept ID 10 两次。于是其计数 c=2, 其他 ID 未出现在历史中。

对出现在窗口内的 token, 当前实现为:

```text
logit <= 0: adjusted = logit * repeat_penalty
logit >  0: adjusted = logit / repeat_penalty
adjusted -= count * frequency_penalty + presence_penalty
```

ID 10 的 logit 为负, 得到:

```text
-0.597837001 * 1.2 - 2 * 0.4 - 0.1 = -1.617404401
```

| ID | 新 logit | 重新 softmax 的数学概率 | 分数名次 |
| --- | ---: | ---: | ---: |
| 10 | -1.617404401 | 0.305997909 | 2 |
| 20 | -1.386294361 | 0.385556717 | 1 |
| 30 | -1.897119985 | 0.231334030 | 3 |
| 40 | -2.995732274 | 0.077111343 | 4 |

penalties apply 本身没有按名次重排数组: 原数组 ID 顺序仍可为 `[10,20,30,40]`, `size=4, sorted=false`, p 仍未更新。若接 greedy, 此时 `selected=1`, 选中 ID 20; 若接 `top-k(k=2)`, 有效视图按分数保留 `[20,10]`。

presence 只按“出现过”减一次 0.1, frequency 按两次历史减 0.8。repeat penalty 也不是每出现一次再乘一次; 这里仅按是否在窗口中应用一次乘法。只有先 accept, 计数才存在; 对同一副本反复 apply 会叠加 logit 变换, 不能当成独立重复实验。

## 6. p 字段、entropy 和有效候选

不能在任意阶段直接用 `p` 计算 entropy。原始候选的 p 是 0; top-p 后 p 可能只剩部分质量; temperature/penalties 后 p 还可能与 logits 不一致。

观测时在局部副本上计算:

```text
m = max(有限 logits)
q_i = exp(logit_i - m) / sum_j exp(logit_j - m)
H = -sum_i q_i * ln(q_i), 单位 nat
```

`-INFINITY` 对应 q=0, 按约定不贡献熵。当前数学推演中 A 最终 H 约为 `0.652860131`, B 为 `0.457835720`。这不是对某个中间步骤旧 p 的直接求和。

还应分别记录 `size` 和 `finite_logits`: grammar 常把非法候选变成 `-INFINITY` 而不缩小 size。空候选、全屏蔽、NaN、正无穷应直接诊断, 不计算伪概率或继续交给 dist。

## 7. greedy 与固定 seed 的复现条件

| 方式 | 需要固定的条件 | 结论边界 |
| --- | --- | --- |
| greedy | 同一候选 ID/logits、历史、grammar 状态、预处理顺序、并列项顺序 | 不依赖 seed; 并列最大值或数值误差仍可能改变选中 ID |
| 固定 seed 的 dist | 上述条件, 加同一初始 RNG、相同 apply/sample 次数与分支、相同 sampler 参数和实现环境 | seed 只是 RNG 起点, 不能替代当前状态 |
| 真实模型整段生成 | 额外固定模型/模板/prompt IDs、context/memory、计算后端与相关数值配置 | 同 seed 不保证跨平台或不同推理配置逐 token 相同 |

对正且有限的 T, 除以 T 保持严格大小关系, 所以单独缩放通常不改变最大 logit 的 ID。它仍会改变随机分布及 top-p 的保留集合。极端浮点溢出、NaN 和并列项需另处理。

`T<=0` 不是做除零或负温度概率变换: 当前 `llama_sampler_temp_impl()` 保留一个最大 logit, 其他设为 `-INFINITY`, 但不会替调用者填写 selected。最终仍需选择器, 或显式使用 greedy。temp=0 后接 dist 与直接 greedy 的 RNG 消耗也不一定相同。

## 8. CSV 与手动验收状态

计划实测文件为 `/home/king/llama-study/results/week6/sampler-stages.csv`。本文的两张阶段表可作为填表参照, 但不冒充 C++ 实测 CSV。建议表头:

```csv
path,step,stage,size,finite_logits,sorted,selected,selected_id,top_ids,logits,p_fields,p_sum,entropy,seed,accepted_history,evidence
```

实际 GDB 需在 penalties/top-k/top-p/temp/dist 的入口及返回后比较 data 指针和有效范围。推演表中的随机 selected 未填, 因为没有运行 C++ RNG, 不会指定一个虚构的“seed=42 抽样结果”。

- [x] 已给出 A/B 的保留 ID、最终概率及中间 p 的有效性。
- [x] 已给出重复历史、penalties、top-k 和 greedy 的数值例子。
- [x] 已解释顺序影响与复现条件, 算术已经独立复核。
- [ ] `test-sampling`/GDB/实际采样的阶段记录及 CSV: 按要求暂时跳过。

源码入口: [llama-sampler.cpp](../../../../src/llama-sampler.cpp) 的 `llama_sampler_temp_impl/softmax_impl/top_k_impl` 及 penalties/top-p/greedy/dist 的 apply、[test-sampling.cpp](../../../../tests/test-sampling.cpp) 的 `sampler_tester`。
