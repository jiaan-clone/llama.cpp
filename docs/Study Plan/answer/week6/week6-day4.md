# week6 day4: Grammar 状态、候选屏蔽与完整输出

本文回答[第 6 周学习路线](../../006-llama-cpp-week6-daily-learning-route.md)中 Day 4 的当天输出与验收。

核对基准为当前 checkout `a8741cc99dadd9bea67ad91c225b69ce772909f6`。以下限定 host sampling、非 lazy grammar。枚举例子和断点表均为源码推导; validator 执行、真实词表候选测试和 GDB 手动实验按要求暂时跳过。

## 1. Grammar 对象和状态图

```text
GBNF 文本 + root 名 + vocab
  -> llama_grammar_parser::parse()
  -> 检查规则、root、左递归等
  -> rules: 持有规则元素
  -> stacks: 各条仍可匹配的解析路径
       每个 stack 保存指向 rules 内元素的指针
  -> partial_utf8: 已接受前缀末尾尚未完成的 UTF-8 字符

当前状态 S = (rules, stacks, partial_utf8)
  +-- apply(candidates)
  |     按各候选 piece 暂时试匹配
  |     不合法 -> logit=-INFINITY
  |     S 不提交前进
  |
  +-- 选择 token t -> accept(t)
        消费 t 的完整 piece, 更新 stacks 和 partial_utf8
        得到下一状态 S'
```

`rules` 是固定语法内容; `stacks` 是“到目前为止还可能成功的解析路径集合”。它不是生成 token 的历史列表。clone 要把栈内指针重新指向克隆后的 rules, 不能仅复制地址。

| 状态 | 含义 |
| --- | --- |
| `stacks` 非空, 所有 stack 均非空 | 当前前缀仍有完成的可能, 但尚不能结束 |
| `stacks` 中至少存在一个空 stack | 至少一条解析路径已完整匹配, 可以结束; 其他非空路径也可能继续 |
| `stacks` 集合为空 | 没有可行解析路径, 不是“已经匹配完” |

## 2. 枚举 grammar 与正反例

`decision.gbnf` 的内容草案为:

```gbnf
root ::= "accept" | "revise"
```

它不允许额外空格或换行。规则文件自身的末尾换行是 GBNF 源码排版, 输入文本末尾的换行则是待匹配字符, 两者不同。

```text
起点: 等待 accept 或 revise, 不允许 EOG
  +-- accept 'a' -> 等待 ccept
  |     -> accept 'c' -> 等待 cept
  |     -> ... -> accept 't' -> 存在空 stack, 允许 EOG
  |
  +-- accept 'r' -> 等待 evise
        -> ... -> accept 'e' -> 存在空 stack, 允许 EOG

不匹配任何路径的字符 -> 无存活路径
```

图中按字符书写只是帮助理解。实际一次 accept 的 token piece 可能一次跨过多个字符, 例如假设词表有 piece `accept`, 它能一次完成整个分支。

| 输入 | 是否可作为当前前缀 | 是否完整匹配 | 源码预测的完整验证结果 |
| --- | --- | --- | --- |
| 空串 | 是 | 否 | Unexpected end of input, 位置 0 |
| `a` | 是 | 否 | Unexpected end of input, 位置 1 |
| `ac` | 是 | 否 | Unexpected end of input, 位置 2 |
| `accept` | 是 | 是 | valid |
| `revise` | 是 | 是 | valid |
| `accept!` | 否 | 否 | Unexpected character `!`, 位置 6 |
| `Accept` | 否 | 否 | 首字符 `A` 不匹配, 位置 0 |
| `accept` 后跟 LF | 否 | 否 | 位置 6 的换行不在 grammar 中 |
| `acceptrevise` | 否 | 否 | 只能选择一个分支, 位置 6 的 `r` 非法 |

位置是此处 ASCII 输入的零基位置, 不把这个简单例子推广为工具对所有 Unicode 显示偏移的保证。表中结果未执行验证器, 不是命令 stdout 的实测摘录。

当前 `tests/test-gbnf-validator.cpp` 会逐字符推进状态, 最后查找空 stack。它对验证失败打印 `Input string is invalid according to the grammar.` 及原因, 但执行到末尾仍返回 0。自动化断言不能只检查退出码, 必须检查输出文本或直接使用验证结果。此工具不使用实际词表, 不能替代 token 级验证。

## 3. apply 如何检查 token

`llama_grammar_apply_impl()` 的主要过程是:

1. 检查 stacks 是否有空 stack, 计算 `allow_eog`。
2. 遍历候选 ID, 从词表获取实际 piece。
3. 对 EOG, 若不允许结束则设为 `-INFINITY`; 对空 piece 或首字节为 0 的普通候选直接屏蔽。
4. 对其余 piece, 结合当前 `partial_utf8` 解码完整码点及剩余部分, 再按规则集合检查。
5. 对没有可行路径的候选把 logit 设为 `-INFINITY`。

本函数不压缩 size、不填归一化 p、不填写 selected, 也不提交 token。当前实现还不主动更新 sorted 标志; 若前面已经按旧 logits 排序, 屏蔽后排序假设可能失效。因此本周把 grammar 放在建立排序/截断的策略之前, 从 `sorted=false` 的原始视图开始。

以 `ac` 为已接受前缀, 以下是候选 piece 的逻辑示意, 未假定实际词表一定有这些单 token:

| 候选 piece / 类型 | 是否允许 | 原因 |
| --- | --- | --- |
| `c` | 是 | 变成 `acc`, 可继续 |
| `cept` | 是 | 变成 `accept`, 完成匹配 |
| `cept!` | 否 | 必须检查整个 piece, 末尾 `!` 非法 |
| `r` | 否 | `acr` 不属于任何合法前缀 |
| EOG | 否 | `ac` 尚未完整 |

当已接受前缀是 `accept` 时 EOG 可以通过, 普通追加 `!` 不可以。允许 EOG 不等于程序一定会选中它; 应用必须继续采样或按自己的终止协议检查完整性。

### UTF-8 与显式 token 规则

对于包含中文的字符 grammar, 一个候选可能只贡献某个 UTF-8 字符的前几个字节。如果仍可能补全成允许的码点, 可以暂时保留, accept 后把未完成部分写入 `partial_utf8`。后续 piece 再补全。枚举 `accept/revise` 本身只有 ASCII, 不能用它的测试结果证明跨 token UTF-8 路径正确。

当前 GBNF 也支持显式 token 规则, 如 `<[1000]>`、按 token 文本引用及取反。字符规则依据 piece 的码点序列, token 规则直接检查 token ID, 不能把当前 grammar 描述成“只能逐字符匹配”。本周枚举与 JSON 草案使用普通字符规则。

## 4. common 的两条路径

以下省略本周关闭的 reasoning budget 处理, 从原始模型 logits 开始。

```text
grammar_first=true
  set_logits: 重建候选, selected=-1, sorted=false
    -> grammar.apply: 屏蔽非法候选
    -> chain.apply: penalties / 温度 / 截断 / 最终选择
    -> 返回 ID, 尚未 common accept
    -> 调用者 common_sampler_accept(id,true)
```

```text
grammar_first=false (默认)
  set_logits -> chain.apply -> 候选 ID t
    -> 为 t 构造单候选 {id=t, logit=1, p=0}
    -> grammar.apply(单候选)
         +-- 未屏蔽: 返回 t
         |
         +-- 被屏蔽:
               set_logits: 从模型 logits 重建全部候选
               -> grammar.apply
               -> chain.apply: 再选择一个合法 ID
               -> 返回 ID
    -> 调用者 common_sampler_accept(id,true) 一次
```

单 token 检查只调用 grammar apply, 不调用 grammar accept, 所以不会把失败候选提交到 grammar。重采样必须从原始 logits 重建候选, 不能继续使用第一次 top-k/top-p 已截断的数组。

第一次 chain 可能已经消耗随机数; 失败时第二次 chain 再消耗。因此相同 seed 下两条路径不保证产生相同 token 序列。即使只看最终分布, 遇到 top-k/top-p 等截断也不能一般性地认为等价。例如先 top-k 保留 A/B、grammar 仅允许 B/C: 默认路径初次抽中 B 会直接返回; grammar-first 则可能让 C 进入 top-k, 获得不同概率。

## 5. 本周自建链的顺序

```text
penalties -> grammar -> temperature -> top-k -> top-p -> dist
```

先用历史调整分数, 再排除不合法候选, 最后在合法集合上控制温度、截断和抽样。比如 logits 中 A/B 分数最高但均非法, C/D 分数较低且合法: 若先 `top-k(k=2)` 再 grammar, 剩下 A/B 会被全部屏蔽; 若先 grammar 再 top-k, 可以保留 C/D。

common 的 grammar-first 是 `grammar -> chain`, 不等于上述每个节点顺序完全相同。若 chain 内包含会改变候选或特殊状态的其他策略, 要按实际配置记录, 不用“都有 grammar”代替顺序证明。无合法有限 logit 时应报告异常, 不能把全 `-INFINITY` 交给 dist 期待它自行恢复。

## 6. 断点记录模板与输出完整性

本次未启动 GDB, 下表仅规定后续观察点, 不包含虚构的命中次数、候选 ID 或内存地址。

| 断点/位置 | 应记录的数据 | 需要验证的事实 |
| --- | --- | --- |
| `llama_sampler_grammar_apply` 入口/返回 | data 指针、size、finite_logits、sorted、selected | 通常 size 不变, 非法项成为负无穷 |
| `llama_grammar_apply_impl` 的 EOG 分支 | stacks 的空栈情况、当前候选是否 EOG | 不完整前缀不能直接结束 |
| `llama_grammar_accept_impl` 前后 | 选中 ID、piece hex、stacks、partial_utf8 | 仅接受最终 token 后提交状态 |
| `common_sampler_sample` 的重采样处 | 两次候选构造、chain apply 次数 | 非法 ID 未被 accept, 重建视图后再采样 |

**为什么合法前缀在 token 上限处被截断, 仍可能不是完整输出?** Grammar 保证每个已选 token 可继续, 但 token 预算可能在 `ac`、`{"goal":"rev` 或尚未闭合的数组处耗尽。存在后续合法延伸不等于当前就能结束。应分别检查终止原因、完整 JSON 解析、Schema 与业务结果。即使预算恰好停在一个完整对象后, 若验收协议要求正常 EOG 结束, 也要单独报告 budget stop。

- [x] 已给出 rules/stacks/partial_utf8 状态图、枚举正反例和两条 common 路径。
- [x] 已解释候选屏蔽、EOG、grammar 与概率策略顺序及前缀/完整输出区别。
- [ ] `decision.gbnf` 等独立实验文件、validator 输出、真实 token/UTF-8/EOG 测试与断点记录: 按要求暂时跳过, 本文仅保留草案和预期。

源码入口: [GBNF Guide](../../../../grammars/README.md)、[llama-grammar.h](../../../../src/llama-grammar.h)、[llama-grammar.cpp](../../../../src/llama-grammar.cpp) 的 parser/apply/accept/clone 与 reject-candidates、[llama-sampler.cpp](../../../../src/llama-sampler.cpp) 的 grammar sampler、[sampling.cpp](../../../../common/sampling.cpp) 的 `common_sampler_sample()`、[test-gbnf-validator.cpp](../../../../tests/test-gbnf-validator.cpp)。
