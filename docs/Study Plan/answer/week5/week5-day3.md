# week5 day3: sequence 操作与 context shift

本文回答[第 5 周学习路线](../../005-llama-cpp-week5-daily-learning-route.md)中 Day 3 的当天输出与验收。以当前 checkout `a8741cc99` 为准, 与路线标注的 `545ddb30a` 不同。

以下均为源码分析和纸面推演, 采用普通单 position、causal attention、unified KV、F16 K/V 的范围。五类操作分别从各自图中的初始状态出发, 不是依次执行的一组命令。手动模型实验和 GDB 暂时跳过。

## 1. 范围、返回值与约束

表中简称 `seq_*` 均对应公共 `llama_memory_seq_*` API, 第一参数为 `llama_get_memory(ctx)`。

| 操作 | 选择范围与效果 | 返回值 / 边界 |
| --- | --- | --- |
| `seq_cp(src,dst,p0,p1)` | 复制源 sequence 的状态关联; 普通 KV 同 stream 对范围内 cells 增加 dst owner | void; 本题 dst 为空, src/dst 为合法 ID; 跨 stream 不是同一种代价 |
| `seq_rm(seq,p0,p1)` | 删除范围内的指定 owner; 最后一个 owner 消失才使 cell 可复用 | bool; 必须检查; 普通 KV 正常路径返回 true, recurrent/hybrid 可能拒绝部分删除 |
| `seq_keep(seq)` | 当前普通 KV 实现遍历目标所属 stream, 清除其他 owner 和无关 cells | void; 本题 unified 只有一个 stream, 不推广为所有 stream 的全局清理 |
| `seq_add(seq,p0,p1,delta)` | 修改选中 cell 的 position, 累积 shift | void; 普通 KV 要求单 position 表示; delta=0 不改变状态 |
| `seq_div(seq,p0,p1,d)` | 对选中 position 整数除以 d, 累积位置修正 | void; 按公共约定使用 `d>1`, 不传 0 或负数; 实现对 d=1 直接返回 |

普通 KV 的带范围操作采用半开区间 `[p0,p1)`: 负 p0 规范化为 0, 负 p1 规范化为最大 `llama_pos`, 用作无上界约定。这里的区间针对逻辑 position, 不是物理 cell 下标。

只有 `seq_rm` 提供匹配全部 sequence 的负 ID 语义; 当前 KV 实现接受的通配值是 `-1`。不要把 `-1` 随意传给其他要求有效 sequence ID 的操作。所有位置查询只是 metadata 观察, 不能仅凭 min/max 证明 cell 共享、总占用或所有位置都无空洞。

## 2. 五类操作的独立前后状态图

记 `cN: pK,{owners}` 为物理 cell N 保存 position K、归属 owners。`empty` 表示 `pos=-1`、无 owner。

### 2.1 `seq_cp(0,1,0,2)`: 同 stream 共享前缀

```text
before: seq 1 为空                     after:
c0: p0,{0}                    ->      c0: p0,{0,1}
c1: p1,{0}                    ->      c1: p1,{0,1}
c2: p2,{0}                    ->      c2: p2,{0}
used = 3                              used = 3
```

同 stream 分支只调用 cell 的 `seq_add`, 不复制各层 K/V tensor, 不产生 logits。因为区间右端不包含 2, c2 不进入 seq 1。后续给两条 sequence 分别 decode 私有 token, 才会分配各自的私有 cells。

共享的前提是模型、前缀 token、位置和相关计算条件相容; 这个 API 不替调用者验证两段文本是否真是同一前缀。

### 2.2 `seq_rm(0,0,-1)`: 删除 owner, 保留其他分支

```text
before:                               after:
c0: p0,{0,1}                  ->      c0: p0,{1}
c1: p1,{0,1}                  ->      c1: p1,{1}
c2: p2,{0}                    ->      c2: empty
c3: p2,{1}                    ->      c3: p2,{1}
used = 4                              used = 3
```

c0/c1 仍属于 seq 1, 因而继续占用物理存储。c2 归属清空后, metadata 将它标为空槽, 并可能调整分配 head, 让以后输入优先复用。旧 K/V 字节不必立即清零; 后续计算依据有效 metadata 使用 cache。

### 2.3 `seq_keep(0)`: 只保留目标分支

```text
before:                               after:
c0: p0,{0,1}                  ->      c0: p0,{0}
c1: p1,{0,1}                  ->      c1: p1,{0}
c2: p2,{0}                    ->      c2: p2,{0}
c3: p2,{1}                    ->      c3: empty
used = 4                              used = 3
```

保留 seq 0 的 cells 时, 还会移除它们的其他 owner。这个结果依赖本题 single stream: 当前 `llama_kv_cache::seq_keep()` 只遍历 `v_cells[seq_to_stream[seq_id]]`, 不能声称它还清理了其他独立 stream。

### 2.4 `seq_add(0,5,8,-3)`: 单 sequence 的位置移动

下面假设 p2..p4 已在独立的前置删除中移除, 存活部分全部属于 seq 0:

```text
before:                               after metadata update:
c0: p0,{0}, shift=0           ->      c0: p0,{0}, shift=0
c1: p1,{0}, shift=0           ->      c1: p1,{0}, shift=0
c5: p5,{0}, shift=0           ->      c5: p2,{0}, shift=-3
c6: p6,{0}, shift=0           ->      c6: p3,{0}, shift=-3
c7: p7,{0}, shift=0           ->      c7: p4,{0}, shift=-3
used = 5                              used = 5
```

物理下标 c5/c6/c7 没有整体搬到 c2/c3/c4。`pos_add()` 更新各 owner 的位置统计并记录 pending shift。若新 position 变为负数, 当前 KV 实现会清空该 cell 的归属并释放其槽位; 这与本图只移动仍有效的位置不同。

**共享限制:** position 是 cell 的共同字段。例如 `c0:p0,{0,1}` 被以 seq 0 选中后加 1, 会成为 `c0:p1,{0,1}`, 不是 `seq0:p1,seq1:p0` 两份状态。API 不会自动拆出隔离副本。单独移动共享前缀中一个 owner 会影响另一个 owner, 所以本周 shift 用单 sequence。

### 2.5 `seq_div(0,0,4,2)`: 压缩 position 编号

```text
before:                               after metadata update:
c0: p0,{0}                    ->      c0: p0,{0}
c1: p1,{0}                    ->      c1: p0,{0}
c2: p2,{0}                    ->      c2: p1,{0}
c3: p3,{0}                    ->      c3: p1,{0}
used = 4                              used = 4
```

整数除法会产生重复 position, 不合并对应 cells, 不释放 buffer, 也不自动扩容。`seq_pos` 使用计数 map, 可以表示 `p0` 和 `p1` 各出现两次。

源码细节应按当前实现记录: `llama_kv_cells::pos_div()` 执行 `pos /= d` 后采用 `shift += p_old - pos` 并设置 `has_shift=true`。本图若原 shift 为 0, pending shift 为 `0,1,1,2`。它与 `pos_add()` 直接累加 delta 的写法不同, 不应在笔记中擅自改写为统一的 `new_pos-old_pos` 公式。本文只核对实际 metadata 代码, 不把未运行的 div 数值计算写成已验证的 RoPE 等价性。

## 3. 同 stream 与跨 stream copy

| 项目 | 同 stream | 跨 stream |
| --- | --- | --- |
| metadata | 源 cells 增加 dst 归属 | 先重置目标 stream metadata, 按源序列建立目标 cell 状态 |
| K/V 数据 | 继续共享同一份 tensor 数据 | 向 `sc_info.ssrc/sdst` 排队, 后续实际复制各层 stream buffer |
| 范围 | 可选择位置区间 | 当前分支要求完整 KV buffer 范围, 不能当任意子区间 copy 使用 |
| 完成时刻 | owner 关联即时改变 | `seq_cp()` 返回时仍可能有待执行的 tensor copy |
| 本周使用方式 | unified、空 dst 的共享前缀 | 仅源码对照, 不执行跨 stream 实验 |

跨 stream 的后续路径为 `init_update -> llama_kv_cache_context::apply -> update`, 其中先同步, 再通过 `ggml_backend_tensor_copy` 复制各层 K/V。这有真实数据复制成本, 与给 bitset 增加一个 owner 不同。

## 4. context shift 的完整手算与两阶段图

初始单 sequence 0 保存 positions `0..7`, `n_past=8`, 保留前 2 个 token, 丢弃接下来 3 个:

```text
llama_memory_seq_rm (mem, 0, 2, 5)
llama_memory_seq_add(mem, 0, 5, 8, -3)
n_past = 5
next_pos = 5
```

实际程序必须先检查 `llama_memory_can_shift(mem)`, 检查 `seq_rm` 的 bool 返回值, 删除失败时不能继续假装已经腾出空间。

| 阶段 | 有效逻辑 positions | used cells | pending shift | 应用侧 next position |
| --- | --- | ---: | --- | ---: |
| 初始 | `0,1,2,3,4,5,6,7` | 8 | 无 | 8 |
| 删除 `[2,5)` | `0,1,5,6,7` | 5 | 无 | 仍待应用层调整 |
| `seq_add(...,-3)` | `0,1,2,3,4` | 5 | 原 p5/p6/p7 的 cells 为 -3 | 应设为 5 |
| 后续 update 完成 | `0,1,2,3,4` | 5 | 已处理并重置 | 5 |
| 新 token 在 p5 decode 完成 | `0,1,2,3,4,5` | 6 | 无 | 6 |

```text
阶段 A: metadata
  seq_rm: 移除 p2,p3,p4, used 8 -> 5
  seq_add: 存活后缀 p5,p6,p7 -> p2,p3,p4
           记录 shift=-3, K tensor 尚未完成修正
  application: n_past / next_pos 8 -> 5

阶段 B: 已有 K 的计算更新
  下一次 decode -> memory_update(false)
    -> init_update(do_shift=true)
    -> update -> build_graph_shift
    -> k_shift 输入 -> build_rope_shift -> graph_compute
    -> reset_shift; 同步后确认数值完成
  再对新输入 p5 执行普通前向计算并生成新 logits
```

普通 RoPE 路径要修正 K 的位置旋转, V 不需要相同的 RoPE 修正。若模型没有 RoPE, 当前实现不会为 shift 构造这类 K 旋转图。存活状态的物理下标不需要因位置变化整体搬移。

删除带来容量回收, shift 使存活后缀和应用 next position 重新协调。只更新 memory 却继续用旧 `n_past=8` 提交输入, 不能得到预期的连续位置。

[`completion.cpp`](../../../../tools/completion/completion.cpp) 的普通上下文移动分支使用 `n_left=n_past-n_keep`, `n_discard=n_left/2`, 随后执行上述删除与负移, 并 `n_past -= n_discard`。本例 `n_past=8,n_keep=2` 得到 `n_discard=3`。

## 5. 回收、能力判断与 cache full 的边界

`seq_rm()` 涉及三个不同概念:

1. **归属删除:** 从某 cell 的 owner 集合中去掉一个 sequence, 共享 cell 可以继续有效。
2. **槽位回收:** 最后一个 owner 消失后将 cell 标为空, allocator 可以重用这个位置。
3. **物理 buffer 释放:** 归还 backend 存储是另一种生命周期操作; `seq_rm()` 不缩小预分配 tensor。`clear(true)` 清零数据也不等于释放 buffer。

`can_shift=true` 只表示当前 memory/model 支持其定义的位置移动机制。当前普通 KV 对 `STEP35` 和多 position 表示返回 false; recurrent 当前返回 true, 但它主要改变 tail position, 不代表能擦除任意历史影响。hybrid 的判断委托 attention 部分。部分删除和历史回退还必须分别满足对应 memory 的限制。

shift 后保存的深层 K/V 已受旧历史影响, 删除一些 cells 不会逆向重算其他 token 的 hidden state。因此不能要求它与“删除原文本后重新 prefill”逐 logits 相等, 也不能把支持 shift 理解为无限无损上下文。

普通 `decode()` 找不到 slot 时可能返回 `1`; 应用根据任务决定停止、删除已完成序列、在支持时移动上下文, 或从所选历史重新 prefill。core 不负责自动为任意输入保留无限历史。

## 6. 验收与跳过说明

- [x] 已给出 `cp/rm/keep/add/div` 五张独立前后图和区间、共享及 stream 限制。
- [x] 已完成 `0..7` 的 context shift 手算及 metadata/计算两阶段图。
- [x] 已区分归属删除、槽位复用、buffer 释放与 `can_shift` 的能力边界。
- [ ] 单 sequence shift、pending K 更新断点、同步后的 tensor 变化与继续 decode: 按要求暂时跳过; 所有数值表均为纸面预期。

源码入口: [llama.h](../../../../include/llama.h) 的 memory API、[llama-kv-cache.cpp](../../../../src/llama-kv-cache.cpp) 的五类 `seq_*`、`get_can_shift/update/build_graph_shift`、[llama-kv-cells.h](../../../../src/llama-kv-cells.h) 的 `seq_rm/seq_keep/pos_add/pos_div`、[llama-context.cpp](../../../../src/llama-context.cpp) 的 `decode/memory_update`。
