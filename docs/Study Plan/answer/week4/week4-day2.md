# week4 day2

本文解答[第 4 周学习路线](../../004-llama-cpp-week4-daily-learning-route.md)中 Day 2 的“当天输出与验收”: batch 字段与默认值、四个切分手算例子、下标映射、ubatch 布局和实际 splitter 选择。

源码基准为当前工作区 `dd545f7e7`, 与路线记录的 `545ddb30a` 不同。以下以普通单位置编码、无 backend sampling 的路径为主。手算与现有测试源码已经核对, 但没有运行测试、模型或 GDB; 手动实验暂时跳过, 不将测试源码中的断言当作本次运行通过的证据。

## 1. 先区分容量、数量与下标

| 名称 | 含义 | 不应混淆为 |
| --- | --- | --- |
| `cparams.n_batch` | 一次 API 调用允许提交的逻辑 token 数上限 | prompt 总长度、并发用户数 |
| `cparams.n_ubatch` | 单个物理计算批次的 token 数上限配置 | 当前 ubatch 的实际长度 |
| `cparams.n_ctx` / `n_ctx_seq` | context 总容量 / 单 sequence 的 context 范围 | 本次输入数量 |
| `cparams.n_seq_max` | context 的 sequence 配置限制参数 | 当前 ubatch 的 `n_seqs` |
| `batch.n_tokens` | 本次提交的实际输入行数 | 数组分配容量 |
| 原 batch 下标 `i` | token 在本次 API 输入中的行号 | sequence 中的 `pos[i]` |
| ubatch 下标 `j` | token 在当前物理批次中的行号 | 原 batch 下标或输出行号 |

同一个 sequence 可以跨多次 API 调用延续。因此每次调用的 batch 下标都可以重新从 0 开始, 但 position 通常从已有历史之后继续。

## 2. 字段与默认值表

### 2.1 `llama_batch` 的输入语义

| 字段 | 含义与普通输入要求 | 缺省处理 / 边界 |
| --- | --- | --- |
| `n_tokens` | 本次有效输入行数; token 输入和 embedding 输入都用它计数 | `llama_batch_init()` 初始设为 0, 调用者必须填写; encode/decode 拒绝零长度输入 |
| `token` | `n_tokens` 个词表 token id, 常用于 tokenizer 得到的文本 | 普通路径与 `embd` 二选一; 有 token 输入时检查 id 非负且小于词表大小 |
| `embd` | 每个输入位置的一条浮点向量, 共 `n_tokens * n_embd` 个 float | 用于调用者已有输入向量的情况, 如适配后的外部特征; 向量宽度必须符合模型输入要求, 不会由 allocator 自动生成 |
| `pos` | 各输入在其所属 sequence 中的位置, 用于位置编码和 memory 状态 | 为 null 时按每条 sequence 的历史推导, 不是一律使用 batch 下标 |
| `n_seq_id` | 每个输入位置所属的 sequence id 数量 | 为 null 时缺省为每个位置 1 个 id; 自定义 `seq_id` 时应同时提供合法的计数数组 |
| `seq_id` | 每个输入位置所属的 id 列表, 例如 `{0}` 或 `{0,1}` | 为 null 时默认 `{0}`; 一个共享前缀 token 可同时属于多个 sequence |
| `logits` | 每个位置的输出选择标记, 类型为 `int8_t *`; 非零表示请求该位置输出 | `output_all=false` 且为 null 时, 只选择整个 API batch 最后一个位置; `output_all=true` 时选择全部位置 |

`batch.logits` 不是保存浮点 logits 的数组。实际 logits 由 context 输出 buffer 保存, 通过输出 API 读取。ubatch 中对应的标记叫 `output`。

**输入 embeddings 与输出 embeddings 模式是两个概念。** `batch.embd != nullptr` 只表示输入是向量; `cparams.embeddings=true` 才是请求提取 embeddings 的 context 配置。前者本身不意味着所有输入位置都需要输出。

普通生成模式中, 显式给出 `logits` 时保留其非零标记。若 `output_all=true`, 显式数组中仍有零, allocator 会警告并使用内部的全非零标记覆盖这一选择。当前 `decode()` 以 `cparams.embeddings` 决定 `output_all`, `encode()` 则固定传 true。

不请求输出的 token 仍参与所需的模型计算。普通 causal Transformer 中, 它们会提供 attention 所需的 K/V 并写入 KV, 后续 token 才能访问这些历史。输出选择可以减少输出投影等计算, 不会把这些 token 从输入历史中删除。

### 2.2 三个 batch API 与数组所有权

| API / 构造方式 | 实际行为 | 所有权与调用者责任 |
| --- | --- | --- |
| `llama_batch_init(capacity, embd_dim, n_seq_max)` | 在堆上分配输入、position、sequence 和输出标记数组; `embd_dim=0` 分配 token 数组, 非零分配向量数组; 返回的 `n_tokens=0` | 数组元素尚未填成有效输入, 调用者必须填写有效行并设置 `n_tokens`; 使用结束后调用 `llama_batch_free()` |
| `llama_batch_free(batch)` | 释放 batch 内数组, 包括每行 sequence id 存储 | 配对释放 `llama_batch_init()` 分配的 batch; 不用于释放借用 vector / 栈内存的视图 |
| `llama_batch_get_one(tokens,N)` | 直接借用传入 token 指针, 设置 `n_tokens=N`, 其他数组指针为 null | 不分配、不复制 token; 调用者保持输入存储在处理期间有效; 后续由 allocator 补齐 sequence、position、输出标记 |
| 调用者自行组装 `llama_batch` | 用指针描述已有数组 | `llama_batch` 不是带析构的拥有型容器, 数组生命周期由调用者负责 |

`llama_batch_init()` 第三个参数是**每个 token 最多可存放多少个 sequence id**的分配容量, 不是在创建 context, 也不自动为 token 分配 sequence id。

这里以公共 API 的普通 token / embedding 互斥输入约定为主。当前 encode/decode 入口因 MTP 等特殊路径只断言至少一个指针存在, 不能据此推荐普通调用者同时填写两者。

## 3. `llama_batch_allocr::init()` 如何补齐与校验

### 3.1 主要顺序

```text
clear 内部状态
  -> 浅复制 batch_inp 到 allocator 内部 batch
  -> 检查 token id 与显式 sequence id 范围
  -> 补齐 n_seq_id / seq_id
  -> 根据 memory 和 sequence 生成缺省 pos
  -> 生成或调整输出标记
  -> 统计输出数、每条 sequence 的位置、coupled 关系和 sequence sets
  -> 校验位置连续性、顺序及 coupled 历史范围
  -> split_reset(), 准备后续切分
```

补齐后的指针写入 allocator 自己的 batch, 不会替调用者重写原 `llama_batch` 结构。缺省数组由 allocator 的成员 vector 持有。

显式 sequence id 按传给 `init()` 的局部 `n_seq_max` 检查, 要求 `0 <= id < n_seq_max`。当前 context 在 `kv_unified=false` 时传 `cparams.n_seq_max`, 在 true 时传 `LLAMA_MAX_SEQ`。因此不能把此处的局部 id 检查上限无条件等同于 context 的同名配置。

自动补齐不是任意畸形数组的修复器。例如, 当前显式 `seq_id` 的范围检查会读取 `batch.n_seq_id[i]`, 所以提供自定义 `seq_id` 时必须同时提供有效的 `n_seq_id`。数组大小、每行计数和指针也必须彼此一致。

### 3.2 缺省 position 的推导

普通单位置编码下, 每条 sequence 先建立下一可用位置 `p0[s]`:

```text
没有 memory: p0[s] = 0
有 memory:   p0[s] = memory->seq_pos_max(s) + 1

按原 batch 顺序处理每个输入 i:
    first = batch.seq_id[i][0]
    pos[i] = p0[first]
    对该输入所属的每个 sequence s:
        p0[s] = pos[i] + 1
```

例如 seq 0 已有位置 `0..9`, 本次有 3 个新输入且 `pos=nullptr`, 自动生成 `10,11,12`。无 memory 且只有默认 seq 0 时, 则生成 `0,1,2`。

对于 `{0,1}` 这样的共享输入, 一个 token 使用一个共同 position, 再一起推进相关 sequence 的下一位置。它不是对每个所属 sequence 分别生成一份不同位置的 token。

### 3.3 位置与 coupling 校验的边界

- 普通位置编码下, 若 memory 已有历史, 本次该 sequence 的最小 position 必须等于历史最大 position 加 1。
- 每条 sequence 的位置集合不能有空洞, 原 batch 顺序中的 position 不能倒退。例如 `0,1,3` 和 `4,5,6,3` 分别违反这两类条件。
- 当前 allocator 允许相等位置, 测试 `allows_equal_positions_in_seq` 明确接受 `0,0,1`。因此应表述为“不倒退”, 不应说它强制每个输入位置都严格递增 1; 后续 memory / 模型仍有自己的约束。
- 当一个输入属于多个 sequence 时, allocator 将这些 sequence 视为 coupled。若提供 memory, 被关联 sequence 的历史最小、最大 position 必须一致, 已经分岔的历史不能靠本次共享 token 自动合并。
- M-RoPE 使用不同的位置规则, 本日不展开, 不把上述普通位置约束套到所有多模态输入。

### 3.4 多 sequence 的默认输出不是“各取最后一个”

设原 batch 先放 seq 0 的 4 个输入, 再放 seq 1 的 2 个输入。在普通生成模式下:

```text
原 batch 下标: 0 1 2 3 4 5
sequence id:  0 0 0 0 1 1

logits=nullptr 得到:       0 0 0 0 0 1
若想各 sequence 最后输出: 0 0 0 1 0 1  <- 调用者显式填写
```

默认值只选整个 API batch 的最后一行。输出数量由标记的位置数决定, 不是 sequence id 个数。

## 4. 四个切分手算例子

以下 `i` 表示原 batch 下标, `j` 表示当前 ubatch 下标; `t_i` 是符号输入, 不是实际模型的词表 id。假设已成功 `init()`、位置与容量合法, 从 `split_reset()` 后开始, 每个例子持续调用同一种 splitter 直至结束。

现有 `test-batch-alloc.cpp` 多数例子使用二维 embedding 输入, 因为测试中的空词表不能接受正常 token id。它用 `embd[i]=[100*i,100*i+1]` 追踪原 batch 行号; 下面的切分索引与该测试安排一致。

### 4.1 `split_simple(2)`: seq 0 的 5 个输入

原输入全部属于 `{0}`, `pos=[0,1,2,3,4]`, 输出标记 `[0,0,0,0,1]`。

| ubatch | `j -> i` 映射 | position | 输出标记 / 输出数 | `(n_tokens,n_seq_tokens,n_seqs,n_seqs_unq)` |
| --- | --- | --- | --- | --- |
| 0 | `0->0, 1->1` | `[0,1]` | `[0,0]` / 0 | `(2,1,2,1)` |
| 1 | `0->2, 1->3` | `[2,3]` | `[0,0]` / 0 | `(2,1,2,1)` |
| 2 | `0->4` | `[4]` | `[1]` / 1 | `(1,1,1,1)` |

结果为 `2+2+1`, 三个 ubatch 的 `equal_seqs()` 均为 false。虽然只有 sequence 0, 前两个 ubatch 的 `n_seqs` 仍是 2, 因为 simple 布局把每个输入位置当作一个长度为 1 的布局单元。

`out_ids=[4]`, `get_n_used()=5`; 再调用返回 `n_tokens=0`。前两个 ubatch 没有请求输出, 仍需计算其 K/V。对应测试为 `split_simple_chunks`。

### 4.2 `split_equal(8,false,0)`: seq 0 长 4, seq 1 长 2

先明确原输入:

| 原下标 `i` | 输入 | sequence set | position | 输出标记 |
| --- | --- | --- | ---: | ---: |
| 0 | `t_0` | `{0}` | 0 | 0 |
| 1 | `t_1` | `{0}` | 1 | 0 |
| 2 | `t_2` | `{0}` | 2 | 0 |
| 3 | `t_3` | `{0}` | 3 | 1 |
| 4 | `t_4` | `{1}` | 0 | 0 |
| 5 | `t_5` | `{1}` | 1 | 1 |

第一次选择互不重叠的 `{0}`、`{1}` 两个 set, 每轮给每个 set 增加一个输入。seq 1 只剩 2 个, 所以两组各取 2 个后停止, 即使上限 8 尚未用满。最后按 set 拼接, 而不是按轮交错排列。

| ubatch | 原下标列表 | `j -> i` 映射 | position | 输出标记 | 布局 `(n_tokens,n_seq_tokens,n_seqs,n_seqs_unq)` |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0,1,4,5]` | `0->0, 1->1, 2->4, 3->5` | `[0,1,0,1]` | `[0,0,0,1]` | `(4,2,2,2)` |
| 1 | `[2,3]` | `0->2, 1->3` | `[2,3]` | `[0,1]` | `(2,2,1,1)` |

两次 `equal_seqs()` 均为 true。第一次的布局是 `[seq0 的 2 行][seq1 的 2 行]`, 第二次只有 seq 0。再调用返回空 ubatch, 总计使用 6 个输入。对应测试为 `split_equal_unequal_lengths`。

输出下标必须再单独记录:

| 原输出下标 | 所在 ubatch | ubatch 下标 | 按切分遇到顺序存入的输出行 |
| ---: | ---: | ---: | ---: |
| 5 | 0 | 3 | 0 |
| 3 | 1 | 1 | 1 |

因此原 batch 的输出位置顺序是 `[3,5]`, allocator 在切分时积累的 `out_ids` 却是 `[5,3]`。这是后续输出重排的原因, 不是 token 的 position 改变了。

### 4.3 `split_seq(8)`: 三个独立 sequence 各长 2

按 seq 0、seq 1、seq 2 排列原 batch, 各自 position 为 `0,1`, 各自第二行请求输出。

| ubatch | sequence set | 原下标列表 / `j -> i` | position | 输出标记 / 输出数 | 布局 `(n_tokens,n_seq_tokens,n_seqs,n_seqs_unq)` |
| --- | --- | --- | --- | --- | --- |
| 0 | `{0}` | `[0,1]` / `0->0,1->1` | `[0,1]` | `[0,1]` / 1 | `(2,2,1,1)` |
| 1 | `{1}` | `[2,3]` / `0->2,1->3` | `[0,1]` | `[0,1]` / 1 | `(2,2,1,1)` |
| 2 | `{2}` | `[4,5]` / `0->4,1->5` | `[0,1]` | `[0,1]` / 1 | `(2,2,1,1)` |

每个 ubatch 的 `equal_seqs()` 为 true, 只含一个 sequence-set 布局单元。结果为 `2+2+2`, `out_ids=[1,3,5]`; 容量 8 足以容纳全部 6 行, 但该策略仍分三次。对应测试为 `split_seq_per_sequence`。

**`split_seq()` 不要求所有 token 的 id 集合始终完全相同。** 它从第一个未使用位置出发, 后续只接纳当前集合的子集, 每接纳一次就将当前集合更新为新集合。比如依次为 `{0,1}`、`{0}`、`{1}`, 且 position 为 `0,1,1`: 第一次可以取前两行, 集合缩为 `{0}` 后不能再接纳 `{1}`, 后者留待下一次。这是“逐步缩小”的规则, 不是任意重叠都能合并。

### 4.4 `split_equal(4,*,0)`: 两个输入都属于 `{0,1}`

原下标 `[0,1]`, position `[0,1]`, 每个输入的 `n_seq_id=2`, 输出标记 `[0,1]`。

| 调用 | 返回结果 | 原因 |
| --- | --- | --- |
| `split_equal(4,true,0)` | 空 ubatch, `n_tokens=0` | `sequential=true` 且 `has_cpl=true`, 函数直接报错并返回; 此时尚未消费输入 |
| 随后 `split_equal(4,false,0)` | 取原下标 `[0,1]`, `j->i` 为 `0->0,1->1` | 允许 coupled 输入, 两行组成一个 `{0,1}` set |

成功结果为:

```text
equal_seqs() = true
n_tokens     = 2
n_seq_tokens = 2
n_seqs       = 1
n_seqs_unq   = 2
seq_id_unq   = [0,1]
seq_idx[0]   = 0
seq_idx[1]   = 1
out_ids      = [1]
```

一个输入属于两个 sequence 不等于输入行数翻倍。标记的最后一行也是一个输出位置, 不是自动生成两个独立输出行。对应测试为 `split_equal_coupled`。

此处空 ubatch 表示“该策略拒绝当前输入”, 与“已经消费完全部输入”不同。实际 memory 调用者还会用 `get_n_used() < get_n_tokens()` 检查是否切分失败。

## 5. ubatch 布局与数组生命周期

### 5.1 `n_tokens = n_seq_tokens * n_seqs` 的意义

它表达当前 ubatch 的布局分组关系。对于等长分组, 每组有 `n_seq_tokens` 行, 一共 `n_seqs` 组, 行按组连续排列。`split_equal` 的第二例首个 ubatch 就是 `4=2*2`。

`split_simple` 不建立等长 sequence 分组, 使用 `n_seq_tokens=1,n_seqs=n_tokens,equal_seqs=false` 描述一般布局, 因而乘积等式仍成立, 却不表示有 `n_tokens` 条独立会话。

| 量 | 计数对象 | 反例 |
| --- | --- | --- |
| `n_seqs` | splitter 使用的布局分组数 | simple 的两个 seq 0 输入可得到 `n_seqs=2` |
| `n_seqs_unq` | 当前 ubatch 中实际出现的不同 sequence id 数量 | 两个属于 `{0,1}` 的输入可得到 `n_seqs=1,n_seqs_unq=2` |
| `seq_id_unq` | 不同 id 的列表 | 当前实现按 id 升序建立 |
| `seq_idx[id]` | id 在上述列表中的局部下标 | 未参与当前 ubatch 的 id 对应 -1 |

判断实际涉及多少个不同 id 应使用 `n_seqs_unq`。`n_seq_id[i]` 则回答“第 i 行属于几个 id”, 又是另一种计数。

### 5.2 ubatch 不是简单指向用户数组的切片

三种 splitter 都通过 `ubatch_add(idxs,...)` 构建结果。该函数根据原 batch 下标列表复制 token / embedding、position、输出标记和 sequence id 数据, 建立 `shared_ptr<llama_ubatch::data_t>`, 再让返回结构中的指针指向这份存储。

因此第二例可以重新组织 `[0,1,4,5]` 这样的非连续输入; memory 也能保存多个 ubatch, 而不会因为 allocator 继续切分就让前一个 ubatch 的数组失效。`llama_ubatch` 也允许 `data` 为空的外部非拥有视图, 所以同样不能断言所有 ubatch 都必然拥有数组。

## 6. 实际由谁选择 splitter?

`decode()` 将 `balloc`、有效 `n_ubatch`、输出模式交给 `memory->init_batch()`。memory 类型与布局决定具体调用, API 层没有统一的“总是每 U 行切一刀”。

| memory 路径 | 当前源码的 splitter | 影响 |
| --- | --- | --- |
| 普通 `llama_kv_cache`, `n_stream==1` | `split_simple(n_ubatch)` | 按原输入顺序连续切分; 单 stream 也可能容纳多个 sequence id |
| 普通 `llama_kv_cache`, `n_stream>1` | `split_equal(n_ubatch,true,0)` | 使用等长分组, 要求所选 sequence id 顺序满足规则, 拒绝 coupled 输入 |
| `llama_memory_recurrent`, `embd_all=true` | `split_seq(n_ubatch)` | 按 sequence-set 路径切分 |
| `llama_memory_recurrent`, `embd_all=false` | `split_equal(n_ubatch,true,n_keep_tail)` | 无 rollback 快照时尾部限制为 0; `n_rs_seq>0` 时为 `n_rs_seq+1`, 尾部约束也会改变切分 |

`split_equal` 中的 `sequential=true` 还要求本次选中的独立 sequence id 逐个连续递增, 不是指 token 的 position。Day 2 的第二、第四例直接测试 `sequential=false`, 不应将测试参数原样当作所有真实 KV 路径的调用方式。

若 `N` 为一个合法 API batch 的 token 数, `U>0`, 从未消费状态开始反复使用 `split_simple(U)`, 且 memory 准备成功, 则 ubatch 数为 `ceil(N/U)`。该公式不是所有策略的公式:

- 第二例 `N=6,U=8`, `split_equal` 得到 2 个 ubatch, 而 `ceil(6/8)=1`。
- 第三例同样 `N=6,U=8`, `split_seq` 得到 3 个 ubatch。
- 第四例可以因为策略不兼容而切分失败, 此时不能用向上取整预测成功执行次数。

另外, prompt 总长超过 `n_batch` 时需要应用层安排多次 API 调用; 上述 `N` 不能直接替换为任意长 prompt 的总 token 数。

## 7. 对应测试、验收与跳过说明

| 源码结论 / 手算 | 现有测试名称 | 本次核对范围 |
| --- | --- | --- |
| 默认 seq 0、缺省 position、只输出最后一行 | `autofill_defaults` | 阅读测试输入及断言 |
| 从历史 position 延续、拒绝历史后空洞或重叠 | `pos_from_memory`, `pos_continuity_with_memory` | 阅读 mock memory 和断言 |
| 显式输出标记与全输出模式 | `explicit_logits`, `output_all` | 阅读断言; 全输出覆盖显式零标记的分支另由实现核对 |
| 不连续、倒退与相等 position | `rejects_non_continuous_positions`, `rejects_decreasing_positions`, `allows_equal_positions_in_seq` | 阅读断言 |
| coupled sequence 已分岔时拒绝 | `rejects_coupled_diverged_seqs` | 阅读断言 |
| 四个必做切分例子 | `split_simple_chunks`, `split_equal_unequal_lengths`, `split_seq_per_sequence`, `split_equal_coupled` | 完成纸面索引与布局推导; 第二例 `[5,3]` 的输出顺序还由 `ubatch_add()` 推导 |

当天验收答案为: 已给出字段与默认值表、四个手算例子及 `j->i` 映射; 已用反例区分 `n_seqs` 与 `n_seqs_unq`; 已定位 memory 选择 splitter 的路径并限定 `ceil(N/U)` 的适用条件。

**手动实验暂时跳过:** 未编译或运行 `test-batch-alloc`, 未启用 `LLAMA_BATCH_DEBUG` 收集日志, 未加载模型观察真实切分。后续应保存测试运行结果和实际 ubatch 日志, 再核对这里的预测; 本文不提供虚构的通过数量或运行输出。

## 8. 源码入口

- [公共 batch API](../../../../include/llama.h): `llama_batch`、`llama_batch_init/free/get_one()` 的输入与所有权约定。
- [batch 定义](../../../../src/llama-batch.h)与[实现](../../../../src/llama-batch.cpp): `init()`、`split_reset()`、三种 splitter、`ubatch_add()`。
- [batch allocator 测试](../../../../tests/test-batch-alloc.cpp): `batch_builder`、`mock_memory` 和上述测试用例。
- [普通 KV memory](../../../../src/llama-kv-cache.cpp): `llama_kv_cache::init_batch()`。
- [recurrent memory](../../../../src/llama-memory-recurrent.cpp): `llama_memory_recurrent::init_batch()`。
- [context 实现](../../../../src/llama-context.cpp): `encode()` / `decode()` 向 allocator 传入的 id 上限、输入宽度和输出模式。
