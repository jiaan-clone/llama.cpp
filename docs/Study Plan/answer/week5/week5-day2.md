# week5 day2: KV cell、slot 分配与容量

本文回答[第 5 周学习路线](../../005-llama-cpp-week5-daily-learning-route.md)中 Day 2 的当天输出与验收。源码基准为当前 checkout `a8741cc99`; 路线原标注为 `545ddb30a`。

本文采用普通全层 causal attention、单 stream unified KV、F16 K/V、关闭 Flash Attention 的纸面例子。cell 下标只是示意, 不表示 allocator 的实测选择。手动推理、启动日志采集和 GDB 实验按要求暂时跳过。

## 1. 从 token 到物理存储的映射

```text
输入 batch 第 i 项:
  token ID + position + sequence IDs
       |
       +-- seq_to_stream[seq_id] --------> 存储 stream
       +-- slot_info.strm / idxs --------> 本次选中的物理 cell
                                              |
                                              +-- metadata: pos、seq 集合、shift
                                              +-- layer 0: K / V 对应位置
                                              +-- layer 1: K / V 对应位置
                                              +-- 其余有 KV 的层
```

| 名称 / 字段 | 精确含义 |
| --- | --- |
| token ID | 词表编号; 相同 ID 可以出现在不同位置, cell metadata 本身不保存完整 token 历史 |
| position | 逻辑输入位置, 影响 RoPE、causal mask 等; 不等于 cell index |
| sequence ID | 该状态属于哪些序列, 不是线程 ID 或天然的请求 ID |
| `llama_kv_cells::pos` | 每个物理 cell 的位置, `-1` 表示空槽 |
| sequence bitset | 每个 cell 的 owner 集合, 支持多个 sequence 共享同一份 K/V |
| `shift` / `has_shift` | 待处理的位置修正及其存在标记, 与实际 K tensor 更新分开 |
| `used` | 占用 cell 下标集合; 实现按 `pos != -1` 统计, 不能拿 sequence 长度总和代替 |
| `seq_pos[s][p]` | sequence `s` 在位置 `p` 的计数; 使用计数 map 是因为同一 sequence 可以有重复 position |
| `v_cells` / `v_heads` | 各 stream 的 cell metadata 和下一次分配搜索起点; head 不是逻辑 next position |
| `seq_to_stream` | sequence 到存储分区的映射; unified 时都映射到 stream 0 |
| `layers` | 各层真实的 K/V tensor 与 stream views; 数据和 cell metadata 分离 |
| `slot_info` | 当前 ubatch 的 stream 及物理下标方案, 包括 `s0/s1`、`strm`、`idxs` |

普通构造中 `n_stream = unified ? 1 : n_seq_max`, 每个 stream 有 `kv_size` 个 cells。因此本题普通独立 KV 的总物理 cell 数为 `kv_size * n_stream`; 不能忽略 stream 数。

## 2. 共享前缀与分叉图

两条 sequence 有相同前缀 `t0,t1`, 分别在 position 2 输入 `a2,b2`。即使 `a2` 与 `b2` 的 token ID 偶然相等, 也不能只凭 token ID 认定两份历史状态可合并。

```text
seq 0: c0(t0,p0) -> c1(t1,p1) -> c2(a2,p2)
        ^            ^
        | 共享       | 共享
        v            v
seq 1: c0(t0,p0) -> c1(t1,p1) -> c3(b2,p2)
```

| stream | cell | position | sequence 集合 | 状态 |
| ---: | ---: | ---: | --- | --- |
| 0 | 0 | 0 | `{0,1}` | 一份共享前缀 K/V |
| 0 | 1 | 1 | `{0,1}` | 一份共享前缀 K/V |
| 0 | 2 | 2 | `{0}` | 分支 A 私有 K/V |
| 0 | 3 | 2 | `{1}` | 分支 B 私有 K/V |
| 0 | 4 | -1 | `{}` | 空槽 |

**两条 sequence 各有 3 个位置, 为什么只用 4 个 cells?** 逻辑归属计数是 `3 + 3 = 6`, 物理状态是 `2 个共享前缀 + 1 个 A 私有 + 1 个 B 私有 = 4`。比两个完全独立存储的序列少了 2 份跨层 K/V。

**c2 与 c3 的 position 相同, 为什么不互相参与 attention?** mask 不只比较 position。seq 0 的 query 会检查 `cells.seq_has(cell,0)`, c3 不满足而被屏蔽; seq 1 同理屏蔽 c2。当前普通 causal 路径还屏蔽空 cell 和 `pos_cell > pos_query` 的未来位置。共享前缀因同时拥有两条 sequence 的归属, 对两者可见。

**删除 seq 0 后什么仍存在?** 对 `seq_rm(0,0,-1)` 的纸面结果为:

| cell | 删除后的 position | 删除后的归属 | 是否仍占用 |
| ---: | ---: | --- | --- |
| 0 | 0 | `{1}` | 是 |
| 1 | 1 | `{1}` | 是 |
| 2 | -1 | `{}` | 否, 可供复用 |
| 3 | 2 | `{1}` | 是 |

used cells 从 4 变为 3; seq 1 的位置仍为 `0,1,2`。删除只改归属及空槽 metadata, 不等于立刻擦除所有旧 tensor 字节。

## 3. slot 分配过程

```text
init_batch
  -> split_simple / split_equal: 得到 ubatches
  -> prepare(ubatches)
       -> find_slot(ubatch,false): 只搜索, 产出 slot_info
       -> 保存相关 cells 与 heads
       -> apply_ubatch: 临时占位, 使后续 ubatch 能检查总容量
       -> 遍历完成或失败后, 逆序恢复之前的 metadata
       -> 全部可放置才返回 slot_info 列表
  -> llama_kv_cache_context
       -> process_ubatch -> apply
            -> apply_ubatch: 正式应用当前 ubatch 的 metadata
            -> get_n_kv: 计算 attention 的访问范围
       -> graph set_inputs -> compute: 真正写 K/V
```

`find_slot()` 从对应 stream 的 `v_heads` 开始搜索, 必要时回绕或从头搜索。普通无 SWA 路径主要使用空 cell; SWA 还可能允许复用已不再可见的单 owner cell。本题不利用 SWA 特例。

### 具体规划例子

假设只有 c0/c1 已保存共享前缀, 空槽 c2/c3 可用, 两个私有 token 被分为两个单 token ubatch:

| 时刻 | 操作 | 源码推演结果 |
| --- | --- | --- |
| 规划 A | `find_slot(A,false)` | 例如返回 `strm={0}, idxs={{2}}` |
| 临时占位 A | `prepare -> apply_ubatch` | c2 暂记为 `p2,{0}`, 使规划 B 不会重复使用它 |
| 规划并临时占位 B | `find_slot(B,false)` | 例如选择 c3, 暂记为 `p2,{1}` |
| 规划结束 | 逆序还原 | c2/c3 再次为空, 保留的是两个 slot 方案 |
| 真实计算 A | `mctx->apply` 后执行图 | 先应用 c2 metadata, 随后写入 A 的 K/V |
| 真实计算 B | `next()` 后重复 | c3 同理; 最终成为上一节的布局 |

一次 ubatch 也可能同时包含两条分支, 这不改变“先确认可放置, 再正式应用, 最后图计算”的区分。

## 4. 五种容量不能混淆

| 量 | 含义 | 例子 / 限制 |
| --- | --- | --- |
| 总空闲 cells | 所有可复用物理槽位数量 | 不保证某个 stream 或连续区间足够大 |
| 最大连续空闲区间 | 无断点的空槽长度 | 可能远小于总空闲数量 |
| used cells | 当前占用物理槽位的个数 | 共享前缀图中是 4, 不是 6 |
| `n_kv` | attention 图访问的 KV 范围, 包括 padding 和空洞 | 不等于逻辑 token 数或 used cells |
| 预分配 buffer 字节数 | 各层 tensor 按容量、宽度、类型与布局分配的存储 | 不会因删除几个 owner 自动按比例缩小 |

`find_slot(ubatch,true)` 要求一次找到足够的连续槽位; `false` 可以逐个收集离散下标。当前普通 KV 的 `prepare()` 和单 sequence 状态恢复均传 `false`。例如 8 格布局 `X . X . X . X .`, 有 4 个空槽但最长连续空段为 1; 需要 3 格时, 非连续方案可以取 `{1,3,5}`, 连续方案失败。这只是分配算法示意, 不表示实际 context 允许只分配 8 格。

当前 `get_n_kv()` 对相关 stream 采用以下规则再取最大值:

```text
pad = max(n_pad,256)
n_kv_stream = min(cells.size(), max(pad, PAD(used_max_p1(),pad)))
used_max_p1 = 最大占用物理下标 + 1
```

若 stream 容量为 256 且只有 c0..c3 占用, used 为 4, `n_kv` 仍为 256。若容量 1024、占用下标为 `{0,700}` 且 pad=256, used 为 2, `n_kv` 为 768。mask 决定访问范围内哪些 cell 对 query 有效, 因此空洞不等于有效历史。

## 5. 容量估算与实际日志状态

普通全层 attention、无额外特殊布局时:

```text
KV bytes ~= layers * total_physical_cells
            * (K_width * bytes_K + V_width * bytes_V)

K_width = KV_heads * key_head_dim
V_width = KV_heads * value_head_dim
```

采用学习路线给定的本地 Qwen2 模型参数: 28 层、2 个 KV heads、K/V head dimension 均为 128, F16 每元素 2 bytes。

```text
K_width = V_width = 2 * 128 = 256
每层每 cell: 256*2 + 256*2 = 1024 bytes
跨 28 层每 cell: 28*1024 = 28672 bytes = 28 KiB
256 个物理 cells: 28672*256 = 7340032 bytes = 7 MiB
其中 K = 3.5 MiB, V = 3.5 MiB
```

这些是给定参数下的计算结果, 不是本次 GGUF 解析或运行测量。`deepseek-r1-1.5b-f16.gguf` 在路线中对应 Qwen2 架构, 不能把它标成 Llama 权重实测。

| 待核对项目 | 理论值 / 源码位置 | 实际日志 |
| --- | --- | --- |
| 各层 K/V 数据主体 | 总物理 cells=256 时约 7 MiB | 未采集, 手动实验暂时跳过 |
| K、V 类型和分项大小 | F16, 各约 3.5 MiB | 未采集 |
| 每 stream cells 与 stream 数 | 构造函数打印 `kv_size` 及 `n_seq_max/n_stream` | 未采集; 日志中的 cells 字段不能脱离 stream 数理解 |
| backend buffer 大小 | 构造函数的 `KV buffer size` 日志 | 未采集; 可能包含对齐开销 |

metadata、tensor 描述、对齐、compute buffer 和模型权重均不包含在上述 7 MiB 数据主体估算中。前缀共享节省重复前缀的物理状态占用和重复 prefill, 可让同一份固定容量承载更多分支; 它不保证 allocator 已分配的 backend buffer 或进程 RSS 立即减少。

## 6. 验收与断点证据要求

| 阶段 | 应观察的位置 | 能证明的内容 |
| --- | --- | --- |
| slot 规划 | `find_slot` 返回值; `prepare` 调用栈 | 预定地址、是否容纳全部 ubatch |
| 临时应用 | `prepare -> apply_ubatch` | 容量验证过程, 必须继续观察还原 |
| 正式 metadata 应用 | `process_ubatch -> llama_kv_cache_context::apply` | cell 位置和 owner 已应用, 尚不能证明新 K/V 完成 |
| K/V 写入 | `cpy_k/cpy_v` 对应图执行, 随后同步 | 才能核对新 token 的实际 tensor 数据 |

- [x] cell/sequence 映射、删除后的状态、slot 规划及容量手算已完成。
- [x] 已解释共享节省什么, 以及槽位回收与 buffer 缩小为什么不同。
- [ ] 启动日志、allocator 实际下标、GDB 调用栈与 tensor 读数: 按要求暂时跳过, 上表只列后续证据要求。

源码入口: [llama-kv-cells.h](../../../../src/llama-kv-cells.h) 的 `seq_rm/seq_pos_min/seq_pos_max/pos_add/pos_div`、[llama-kv-cache.h](../../../../src/llama-kv-cache.h) 的 `slot_info`、[llama-kv-cache.cpp](../../../../src/llama-kv-cache.cpp) 的构造函数、`prepare/find_slot/apply_ubatch/get_n_kv/set_input_kq_mask_impl`。
