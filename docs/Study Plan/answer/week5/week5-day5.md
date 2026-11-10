# week5 day5: recurrent/hybrid memory 与能力边界

本文回答[第 5 周学习路线](../../005-llama-cpp-week5-daily-learning-route.md)中 Day 5 的当天输出与验收, 将路线要求的 `notes/week5-day5.md` 内容统一写入本文件。源码基准为当前 checkout `a8741cc99`, 路线原标注为 `545ddb30a`。

普通 KV、recurrent 与 `llama_memory_hybrid` 的对照通过源码完成; 不要求额外模型。周末用例仅给出设计与预期, 模型运行、GDB、回退及恢复实验均按要求暂时跳过。

## 1. 三种状态表示

普通 attention 在计算新 token 的 query 时需要访问历史各位置的 K/V; KV cache 避免再次计算完整前缀。recurrent 则把已消费历史归纳到更新中的状态, 概念上可写作 `state_t = F(state_(t-1),input_t)`, 不按每个历史 token 保留一份普通 attention K/V。

```text
普通 KV:
  seq 0 -> p0 K/V, p1 K/V, p2 K/V, ...
  cell  -> 历史位置的物理槽位, 可有多个 sequence owners

recurrent:
  cells[seq_id].tail -> 状态 cell
                        +-- pos: 已处理到的位置
                        +-- seq_id: owners
                        +-- src/src0: 实际状态来源行
                        +-- r_l/s_l: 各层状态 tensor 的关联行
  n_rs_seq / rs_idx -> 有限回退快照及本次应选用的状态组

hybrid:
  llama_memory_hybrid
    +-- mem_attn: attention 层的 KV cache
    +-- mem_recr: recurrent 层的状态
  两部分共同描述同一批输入的推理状态
```

`r_l` 和 `s_l` 是按层保存的 recurrent 状态 tensor 列表, 宽度来自 `hparams.n_embd_r()` 与 `hparams.n_embd_s()`; 具体数学含义依模型架构而定, 不应将它们简单重命名为普通 K/V。`mem_cell` 同时承担序列 tail 索引和状态 cell 管理, 所以不能把它的下标直接解释成历史 token position。

## 2. 三类 memory 对照表

| 对比项 | 普通 KV | recurrent | hybrid |
| --- | --- | --- | --- |
| 主要数据 | `layers` 中各层 K/V 与 `v_cells` metadata | `r_l/s_l` 及 `mem_cell`、`rs_idx` | `mem_attn` 与 `mem_recr` 两部分, 默认按 `hparams.is_recr(il)` 分配层 |
| cell 含义 | 某个历史位置的物理槽位, owner 可以共享 | 状态槽位及 tail/src 关联, 不等于每个 token 一份历史 K/V | 两种 cell 各自管理, 对同一输入协调更新 |
| 容量主要受什么影响 | 历史物理 cells、层数、K/V 宽度和类型 | 状态槽位数、每层状态宽度、`1+n_rs_seq` 组状态存储 | 同时承担两类存储与约束 |
| `seq_cp` | 同 stream 增加 owner; 跨 stream 排队复制 tensor | 当前 `seq_cp` 将 dst 指向 source tail 并增加 owner; 后续分叉由 `find_slot` 等处理状态来源 | 分别调用 attention 与 recurrent 的 `seq_cp`, 不能只统计一侧代价 |
| 部分删除 / 回退 | 普通路径按 position 区间移除 owner | `seq_rm` 受 tail 与 `n_rs_seq` 约束, 可能 false; 删除中间区间也不等于消除状态中的历史影响 | 先让 recurrent 判断可否删除, 成功后再删 attention |
| position shift | `seq_add/div` 更新 cell position, RoPE K 修正由 update 执行 | `seq_add/div` 主要修改 tail cell 的 position, 不重新计算过去的 `r_l/s_l` | 两侧都执行位置操作; K 更新仍由 attention 侧负责 |
| `get_can_shift()` | 当前排除 STEP35、多 position 表示等不支持配置 | 当前直接返回 true | 当前返回 `mem_attn->get_can_shift()` |
| batch 准备 | `split_simple` 或 `split_equal`, `prepare/find_slot` 验证 cells | `split_seq` 或顺序 `split_equal`, `find_slot` 要求 `equal_seqs()` 并整理连续状态槽位 | 按 recurrent 规则切分, 先 `mem_recr->prepare`, 再 `mem_attn->prepare` |
| pending update | `init_update` 处理 copy/shift | `init_update` 返回 `NO_UPDATE`; 状态计算与复制还可发生在正常前向图 | 组合两侧 status, 统一 apply/next |
| state 读写 | 选中 metadata 与各层 K/V | 选中状态 metadata 与实际 `r_l/s_l` 行 | 默认保存两部分; `PARTIAL_ONLY` 时当前 hybrid 跳过 attention |

对应主要函数分别位于 [llama-kv-cache.cpp](../../../../src/llama-kv-cache.cpp)、[llama-memory-recurrent.cpp](../../../../src/llama-memory-recurrent.cpp)、[llama-memory-hybrid.cpp](../../../../src/llama-memory-hybrid.cpp)。表中判断均限定当前实现, 不代表所有 memory 子类共享同样的数据布局。

## 3. recurrent 的 tail、复制和回退机制

### tail/src 如何参与分叉

`cells[seq_id].tail` 指向该 sequence 当前关联的状态 cell; 状态 cell 的 `seq_id` 集合可以包含多个 owner。`src` 表示应从哪一行取得已有状态, `src0` 为设置图输入保留来源信息。

当前 recurrent 的 `seq_cp()` 会清理已有 dst tail 关联, 将 dst 接到 source tail。虽然签名包含 p0/p1 且会规范化负值, 当前该函数后续没有像 KV 一样逐历史 position 筛选, 因而不能解释成“复制任意历史片段”。

后续 `find_slot()` 发现某序列不独占原 tail 时, 为它准备单独 cell、记录来源, 并整理状态槽位。图输入 `s_copy` 和 `build_rs` 等再用这些索引取出状态。因此一次 metadata 共享操作不能保证之后分叉始终没有真实状态复制成本。

### `n_rs_seq` 不是任意历史的快照库

构造函数为每层 r/s 分配的行数为:

```text
n_rows = mem_size * (1 + n_rs_seq)
```

`n_rs_seq` 为受限的回退状态组提供空间, `rs_idx[seq]` 指示该序列下一次应读取哪一组。`llama_memory_recurrent_context::s_copy()` 在使用回退索引时计算 `idx * size + src0`, 并消费、重置相应索引。实际可用的快照还依赖模型图和本次 ubatch 是否按要求保留状态, 不能仅从参数数值推断任意历史状态都存在。

batch 切分特别要求每条 sequence 末尾的 `1+n_rs_seq` 个 token 留在同一 ubatch, 相关标记为 `TAG_RECURRENT_ROLLBACK_SPLITS`。这也是 memory 必须参与 ubatch 切分的一个直接理由。

## 4. 一组 partial remove 成功与失败条件

假设 seq 0 合法且独占一个有效 tail, `tail.pos=9`, 暂无其他待处理回退。各行都从该初始状态独立推演。

当前 `seq_rm()` 在以下条件下进入尾部回退分支:

```text
0 < p0 <= cell.pos && p1 > cell.pos
rollback = cell.pos - (p0 - 1)

1 <= rollback <= n_rs_seq:
    set_rs_idx(seq,rollback)
    cell.pos = p0 - 1
    return true
否则:
    return false
```

负 p1 先规范化为无上界。下表的成功表示源码允许选择该回退索引, 数值续写正确还需要相应快照有效。

| 初始配置 / 操作 | 计算 | 代码路径的结果 |
| --- | --- | --- |
| `n_rs_seq=0`, `seq_rm(0,9,-1)` | rollback=`9-(9-1)=1` | false, 无法选择前一状态; tail position 保持 9 |
| `n_rs_seq=2`, `seq_rm(0,8,-1)` | rollback=`9-(8-1)=2` | true, `rs_idx[0]=2`, tail position 改为 7 |
| `n_rs_seq=2`, `seq_rm(0,7,-1)` | rollback=`9-(7-1)=3` | false, 超过可选范围; 不修改该 tail |
| `n_rs_seq=2`, `seq_rm(0,0,-1)` | 完整删除, 不走 p0>0 的回退分支 | 合法序列完整删除可成功, 归属及 tail 清理, rollback index 重置 |
| `n_rs_seq=2`, `seq_rm(0,2,5)` | 区间未覆盖 tail position 9 | 对本例可返回 true 但不改变 tail 状态; 不代表已经从 `state_9` 中消除了 token 2..4 的影响 |
| `seq_rm(-1,8,-1)` | 通配所有序列, 非空且非全范围 | 当前 recurrent 分支返回 false |

因此准确结论是: **recurrent 的部分回退受状态表示、保存窗口和输入切分约束, 存在失败情况, 也存在有限可用的回退路径。** “永远不能回退”和“可以任意删除历史”都与当前源码不符。

`get_can_shift()==true` 与这个 bool 删除结果不矛盾。前者只承诺支持其 position 调整机制; 改一个位置整数并不生成 `state_7` 或逆向消除两步 recurrent 更新。

## 5. hybrid 为什么先处理 recurrent

```text
hybrid.seq_rm(seq,p0,p1)
  -> mem_recr->seq_rm(...)
       false -> 直接返回 false, attention 尚未修改
       true  -> mem_attn->seq_rm(...)
```

如果先删除 attention, 再遇到 recurrent 拒绝部分回退, 两部分会描述不同的历史。当前顺序先询问可能失败的一侧, 避免在这种失败条件下提前改坏 attention。这个保证针对这里的删除分支, 不应扩展成所有 memory 操作都有完整事务回滚。

hybrid 的 `init_batch()` 同样需要为同一组 ubatches 同时准备两部分。其 memory context 组合两侧状态, 并在 `apply()`、`next()` 中共同推进。`llama_memory_status_combine()` 遇失败保留失败; 任一侧有更新时组合为 `SUCCESS`, 两侧都无更新才是 `NO_UPDATE`。

### hybrid 和 recurrent 的 state 范围

当前 hybrid 默认依次写 attention 和 recurrent, 读取时按同样顺序恢复。设置 `PARTIAL_ONLY` 时仅处理 recurrent 一侧, 所以这份数据不能独立重建普通完整 hybrid 历史。

recurrent `state_write()` 根据当前 `rs_idx` 选择实际来源行。它保存的是当前选定的逻辑状态, 不能说它把所有备用 rollback 快照原封不动打包。`state_read()` 成功后重置相关 rollback index; 因此也不能要求恢复后还拥有保存前同样的全部回退窗口。共享 recurrent cell 的 owner 若持有不同回退索引, 当前完整 state 写出还有明确的拒绝路径。

## 6. 周末实验预期表

以下所有项目状态均为 **待执行, 本次按要求跳过手动实验**。采用路线约定的 CPU、Qwen2 F16 模型、关闭 Flash Attention、无 LoRA、无 backend sampling、固定 token IDs 和 greedy sampler。多序列用例显式 `n_seq_max=2,kv_unified=true`, 所有配置在前后对照中固定。

| 用例 | 操作 | 预期 | 必须收集的证据 |
| --- | --- | --- | --- |
| 共享分叉 | seq 0 prefill 公共前缀; `seq_cp(0,1,0,P)`; 两分支各 decode 私有输入并请求输出 | cp 后 used 不增加, 前缀 cells owners 为 `{0,1}`; 私有输入占不同 cells; 分支各自与独立完整 prefill 的 greedy 基线一致 | cp 前后 cell/owner 表、used、stream; A/B token IDs; A先/B先两种运行顺序; 正确输出行与 seq ID 映射 |
| 移除隔离 | 保存 seq 1 的 host state, 删除 seq 0, 再保存 seq 1 并继续 seq 1 | 删除返回 true; 共享前缀保留 owner 1; 仅 seq 0 的 cells 可复用; seq 1 state bytes 不变, 受控续写与独立基线一致 | 删除前后序列化字节与长度; cell 归属及 used; seq 1 pending 输入和逐 token 对照 |
| 单序列 shift | seq 0 有 p0..p7; `rm[2,5)`, `add[5,8),-3`; 应用 next_pos=5; decode 新输入 | used 8 -> 5; positions 为 0..4; 后缀 cells 先积累 -3 shift, 随后由 K update 消费; 新输入在 p5 正常计算 | `can_shift`、rm 返回值; metadata/graph 两阶段调用栈; pending shift; 同步后状态; decode 返回值; 不要求与删文本重做 prefill 的 logits 相等 |
| state 恢复 | 保存到 p=P-2, 另存 pending=t[P-1]@P-1; baseline、clear/restore、新 context seq 1 三路继续 | 加载字节数匹配; restored pos_max=P-2; 三路都先 decode 同一个 pending 输入产生新 logits; token IDs 与停止原因一致 | 原始 bytes/长度; 模型与参数标识; clear 后空状态; 恢复位置及目标 seq; pending decode、输出行、greedy token 序列 |

各用例统一记录: 阶段、seq ID、输入数及 positions、stream/cell 范围、used、pos min/max、pending copy/shift、API 返回值或字节数、证据来源。`seq_cp` 没有返回值, 应记录 metadata 变化; min/max 不能替代共享归属观察。

为保证比较有意义, 基线与实验分支使用相同 sampler 和停止规则, 记录 token IDs, 不只比较可见文本。若一次 decode 有多个输出, 先确定各分支对应的输出行, 不假定 `-1` 能同时表示两条 sequence 的最新 logits。recurrent/hybrid 本次以源码对照为必做内容, 不通过下载额外模型扩大实验范围。

## 7. 统一接口究竟统一了什么

统一接口让 `llama_context` 以一致流程准备 batch、应用 memory、推进 ubatch、管理 sequence 和保存恢复状态。模型工厂选择实现后, 上层不必把普通 KV 的数据布局写死在 decode 调度中。

它没有承诺相同的物理结构、相同的 copy 成本、任意区间删除都成功, 也没有承诺 position 改变就等于历史内容重算。使用者仍须检查能力、返回值、目标 sequence 合法性与恢复协议, 并根据具体 memory 的限制设计操作。

## 8. 验收与跳过说明

- [x] 三类 memory 对照表及对应字段/函数已完成。
- [x] 已给出 partial remove 的明确失败例子、有限回退例子及中间删除的语义边界。
- [x] 已解释 hybrid 的删除顺序、batch 协调、shift 能力与 state flags。
- [x] 已准备共享分叉、移除隔离、单序列 shift、state 恢复四组实验预期。
- [ ] 上述用例的模型运行、GDB、实际字节数、token IDs 与退出码: 按要求暂时跳过, 尚无本次实测结果。

补充源码入口: [llama-memory-recurrent.h](../../../../src/llama-memory-recurrent.h) 的 `mem_cell/r_l/s_l/n_rs_seq/rs_idx`、[llama-memory-recurrent.cpp](../../../../src/llama-memory-recurrent.cpp) 的 `seq_rm/init_batch/find_slot/s_copy/state_write/state_read`、[llama-memory-hybrid.cpp](../../../../src/llama-memory-hybrid.cpp) 的 `init_batch/seq_rm/get_can_shift/state_write/state_read`、[llama-memory.cpp](../../../../src/llama-memory.cpp) 的 status 组合逻辑、[llama-graph.cpp](../../../../src/llama-graph.cpp) 的 recurrent 图输入和 `build_rs`。
