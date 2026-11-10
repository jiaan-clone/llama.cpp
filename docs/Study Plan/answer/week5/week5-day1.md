# week5 day1: memory 接口与 decode 的接合点

本文回答[第 5 周学习路线](../../005-llama-cpp-week5-daily-learning-route.md)中 Day 1 的当天输出与验收。

核对基准为当前 checkout `a8741cc99`, 与路线标注的 `545ddb30a` 不同。以下以普通 decoder-only Transformer、CPU、F16 K/V、关闭 Flash Attention、无 LoRA、无 backend sampling 为范围。本文完成源码分析; 模型运行、GDB 和其他手动实验按要求暂时跳过。

## 1. 接口职责图

```text
llama_model
  | create_memory(): 按架构及参数选择实现
  v
llama_context -- owns --> llama_memory_i
  |                        |
  |                        +-- llama_kv_cache: 历史 K/V + cell metadata
  |                        +-- llama_memory_recurrent: recurrent state
  |                        +-- llama_memory_hybrid: attention + recurrent
  |
  +-- balloc: 校验 batch、提供 split_simple / split_equal / split_seq
  +-- sched: 分配和调度 GGML compute buffers
  +-- buf_output: host 可读的 logits / embeddings 等输出

llama_memory_i
  +-- init_batch(...) --> 本次 batch 的 llama_memory_context_i
  |                        +-- ubatches + 本次放置方案
  |                        +-- get_status / get_ubatch / apply / next
  +-- init_full() -----> 图预留使用的满容量 memory context
  +-- init_update(...) -> pending copy/shift 的 memory context
```

`llama_memory_i` 是由 context 持有、跨多次 decode 存活的推理状态对象。`llama_memory_context_i` 是一次 batch、update 或预留过程的控制对象, 引用已有 memory, 并不重新拥有一份完整 KV。普通 KV 实现额外通过 `get_k/get_v/cpy_k/cpy_v`、输入索引和 mask 等方法连接图构建。

这里的 model memory 指模型推理所需的历史状态, 不是 `llama_model` 中的权重。`model.create_memory()` 是工厂入口, 返回对象的所有权交给 `llama_context::memory`。

## 2. 各接口的调用者及返回结果处理

| 接口 | 职责 | 调用者及结果处理位置 |
| --- | --- | --- |
| `init_batch(balloc,n_ubatch,embd_all)` | 切分 ubatch, 验证整个 batch 的放置方案 | `llama_context::decode()` 调用; 检查空指针及 `get_status()`, 成功后逐个处理 ubatch |
| `init_full()` | 提供满容量形状和模拟 slot, 支持最坏情况 compute buffer 预留 | `sched_reserve()`、`memory_update()` 后的 `graph_reserve()` 使用; 不表示已经实际填满 memory, 也不执行正常 token 推理 |
| `init_update(lctx,optimize)` | 收集待执行 copy/shift; 无工作时返回 `NO_UPDATE` | `memory_update()` 检查 status 后调用 `apply()`; 当前普通 KV 忽略 `optimize` 参数, 不能据此假定必然执行碎片整理 |
| `get_ubatch()` | 返回本次 memory context 当前索引对应的输入 | `decode()` 的循环用它统计输出数, 再传给 `process_ubatch()` |
| `next()` | 推进到下一个 ubatch | `decode()` 的 `do ... while (mctx->next())` 结束条件; `false` 表示遍历完成 |
| `apply()` | 普通 batch 应用当前 slot 的 metadata; update context 执行 copy/shift | `process_ubatch()` 在建图前调用; 返回 false 时设置计算失败。`memory_update()` 也调用它并记录失败日志 |
| `get_status()` | 区分准备成功、无更新、准备失败和准备计算失败 | `decode()` 与 `memory_update()` 分别按当前场景处理, 不能把 `NO_UPDATE` 在两个位置的含义混为一谈 |

### status 如何传回上层

| status | `decode()` 中 `init_batch()` 之后 | `memory_update()` 中 |
| --- | --- | --- |
| `SUCCESS` | 开始 ubatch 计算 | 应用 update, 随后重新预留图 |
| `NO_UPDATE` | 属于非预期状态, 返回 `-2` | 正常表示无待处理更新, 返回 false |
| `FAILED_PREPARE` | 至多尝试一次 `memory_update(true)`; 若其返回 true 则重试准备, 否则返回 `1` | 记录准备失败, 返回 false |
| `FAILED_COMPUTE` | 返回 `-2` | 记录准备失败, 返回 false |

还要区分后续实际图计算的状态。`process_ubatch()` 失败后, `decode()` 尝试删除失败 ubatch 对应的位置, 并将 aborted、allocation failed、compute failed 分别映射为 `2`、`-2`、`-3`。这不是“整个 batch 无条件原子回滚”: 已成功的前面 ubatch 与不同 memory 的删除限制仍需考虑。

当前源码有一处需要按实现理解的边界: KV update 分支的 `llama_kv_cache_context::apply()` 调用 `kv->update()` 后直接返回 true, 不转发其结果; `memory_update()` 也不是一个能区分“无工作”和所有失败原因的完备成功标志。观察 update 是否完成, 应结合内部日志、图计算结果和同步后的状态, 不能只看这个 bool。

## 3. 普通 decode 路径

```text
llama_decode(ctx,batch)
  -> llama_context::decode()
       -> balloc->init(): 校验输入、补齐需要的 batch 信息
       -> sched_reserve()
       -> memory_update(false): 处理前面遗留的 copy/shift
       -> memory->init_batch()
            -> split ubatches
            -> KV prepare(): find_slot + 临时 apply + 还原
            -> 返回 ubatches 与 slot_info 列表
       -> 检查 status, 预留输出 buffer
       -> 循环:
            get_ubatch()
            process_ubatch()
              -> mctx->apply(): 当前 ubatch 的 metadata 正式应用
              -> 构建或复用图, 分配 compute buffer
              -> res->set_inputs(): 设置位置、写入索引、mask 等
              -> graph_compute(): 计算本次 K/V 并写入 cache
            -> 提取所请求的 logits / embeddings
            next()
```

`prepare()` 中会为了验证后续 ubatch 是否也放得下, 临时调用 `apply_ubatch()` 然后还原。因此这个函数的断点可能同时出现在规划阶段与真实推理阶段。必须看调用栈: 来自 `prepare()` 的命中不是正式写入完成证据。

### metadata 与 K/V 计算的两个时间点

```text
T0: find_slot 得到物理位置
  -> T1: process_ubatch -> mctx->apply
         cell 已有 pos / seq_id, used cells 已更新
         当前 token 的新 K/V 尚未由这次图计算生成
  -> 建图或复用 -> set_inputs -> graph_compute
  -> T2: 图执行完成并同步
         当前 token 的各层 K/V 已写入, 请求的输出可读取
```

GGML 的 `cpy_k/cpy_v` 在这里创建写入节点, 并不是 C++ 调用到它时就完成数值复制。backend 执行可能异步, 调试时应在 `llama_synchronize(ctx)` 之后判断数据完成, 不把 graph 提交当成所有设备写入已经结束。

## 4. pending update 路径与图输入

```text
seq_add / seq_div
  -> 修改 cell position, 累积 pending shift
跨 stream seq_cp
  -> 准备目标 metadata, 排队 stream copy
  -> 下一次 decode 内的 memory_update(false)
       -> init_update: 取出 do_shift 与 sc_info
       -> apply -> kv->update
            -> 有 copy: 同步后复制各层 stream K/V buffer
            -> 有 shift 且模型有 RoPE:
                 build_graph_shift -> set_inputs -> graph_compute
            -> 成功处理 shift 后 reset_shift
       -> init_full -> graph_reserve
  -> 再开始当前新输入的普通 decode
```

这条路径处理已有状态, 与新 token 的前向计算分开。`init_full()` 使用满容量 `n_kv` 和 dummy slot 信息服务图形状预留, 不应当作可遍历的真实输入 batch 使用。

| 图中用途 | metadata / slot 信息如何进入计算 |
| --- | --- |
| 读取历史 K/V | `get_k/get_v` 按当前 stream 范围与 `n_kv` 创建各层 cache view |
| 写入当前 K/V | `set_input_k_idxs/set_input_v_idxs` 从 `slot_info.strm/idxs` 生成写入地址; `cpy_k/cpy_v` 构造 `ggml_set_rows` 节点 |
| attention 可见性 | `set_input_kq_mask` 检查 cell 是否为空、是否属于 query 的 sequence、position 是否满足 causal 等条件 |
| 位置更新 | pending shift 作为 `k_shift` 输入进入专门的 K 修正图; 普通 RoPE 路径不对 V 做同样的旋转 |

关闭 Flash Attention 时, 当前普通 KV 的 V 可以采用转置布局, V 索引会按维度展开。因此不能假设 K/V 的物理索引表达式始终相同。

## 5. 三类内存分别保存什么

| 类别 | 保存内容 | 生命周期与用途 |
| --- | --- | --- |
| model memory / KV memory | 历史 token 的各层 K/V、位置和 sequence 归属等 | 跨 decode 保留, 供后续 attention 使用; recurrent/hybrid 使用自己的状态表示 |
| GGML compute buffer | 前向计算的中间激活、算子结果等计算空间 | scheduler 按图规划、复用; 它不是长期 token 历史的权威存储 |
| logits output buffer | 已请求输出行的词表分数, 以及按配置保存的 embeddings 等 | context 提供给调用者读取和采样; 后续 decode 可覆盖, 不属于 KV state |

模型权重是第四类资源: 由 model 持有的训练参数, 既不是历史 KV, 也不是本次临时激活。

**为什么 memory 参与 ubatch 切分?** 单纯按 `n_ubatch` 切成等长数组只满足计算大小限制, 不保证有合适槽位或合法状态布局。普通 unified KV 可以 `split_simple`; 多 stream KV 使用适合各 sequence 等量输入的 `split_equal`; recurrent 还要满足状态槽位、相同 sequence 长度和回退快照的边界。memory 知道这些约束, 因而必须参与“这次能计算哪些输入”的决定。

**为什么复制 KV 后不能直接给另一个 sequence 采样?** 复制或共享 memory 没有执行产生新 logits 的前向计算, 也没有建立新输出行与目标 sequence 的对应关系。应按 checkpoint 或分叉协议对正确输入、position、seq ID 执行 decode 并请求输出, 再用该输出采样。

## 6. 验收与跳过说明

- [x] 已给出接口职责图、普通 decode 路径与 pending update 路径。
- [x] 已区分长期 memory、计算 buffer、输出 buffer, 并解释 memory 约束 ubatch 的原因。
- [x] 已从源码区分 slot 规划、metadata 应用、graph 写入与同步完成。
- [ ] 手动 GDB 观察 T1/T2、运行模型并核对实际 K/V 和日志: 按要求暂时跳过, 本文不宣称已取得动态证据。

源码入口: [llama-memory.h](../../../../src/llama-memory.h)、[llama-memory.cpp](../../../../src/llama-memory.cpp)、[llama-context.cpp](../../../../src/llama-context.cpp) 的 `decode/process_ubatch/memory_update/sched_reserve`、[llama-kv-cache.cpp](../../../../src/llama-kv-cache.cpp) 的 `init_batch/prepare/update`、[llama-graph.cpp](../../../../src/llama-graph.cpp) 的 `llm_graph_input_attn_kv::set_input` 与 `build_attn`。
