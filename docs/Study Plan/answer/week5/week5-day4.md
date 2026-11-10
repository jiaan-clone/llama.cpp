# week5 day4: state API、测试断言与恢复边界

本文回答[第 5 周学习路线](../../005-llama-cpp-week5-daily-learning-route.md)中 Day 4 的当天输出与验收。以当前 checkout `a8741cc99` 的实际读写函数为准, 路线标注版本为 `545ddb30a`。

范围为普通 decoder-only Transformer、CPU、F16 K/V、关闭 Flash Attention、无 LoRA、无 backend sampling。本文完成源码与测试断言核对, 未运行模型、state 测试或自建恢复实验; 所需手动实验按要求暂时跳过。

## 1. state API 对照

| API 组 | 实际保存 / 恢复内容 | 返回值及调用者检查 |
| --- | --- | --- |
| `llama_state_get_size` | 用 dummy writer 计算当前 context state 所需字节数 | `size_t`; 保存时取得大小并检查, 不在空恢复目标上重新估计旧状态长度 |
| `llama_state_get_data/set_data` | context 的架构字符串及 memory state | 实际写入 / 读取字节数; 对照保存时的原始长度, 失败返回 0 |
| `llama_state_save_file/load_file` | session magic/version、调用者提供的 token 数及列表、context state | bool; 检查保存和加载结果, 加载时还检查 token 容量和实际返回数量 |
| `llama_state_seq_get_size/get_data/set_data` | 指定 sequence 的 memory state; raw 包装还含内部 magic 和源 seq ID | `size_t`; set 指定合法目标 seq ID, 检查消费字节数和恢复后的 position |
| `llama_state_seq_save_file/load_file` | sequence 文件专用 magic/version、token 数及列表、单 sequence memory state | `size_t`, 成功返回文件写入 / 消费位置, 失败为 0; 不是 bool API |
| `llama_state_seq_*_ext` | 单 sequence API 的 flags 扩展 | 保存和恢复需匹配 flags; `ON_DEVICE` 有额外生命周期依赖 |
| `llama_memory_clear(mem,data)` | 清除 memory metadata; data=true 还清零其数据 buffer | void; 不释放 model, 不重置应用历史或外部 sampler, 也不保证旧 logits 被清除 |

文件与 raw bytes 有各自的包装。完整 context 文件和 sequence 文件不能混用; sequence 文件也不是简单把 `llama_state_seq_get_data()` 的 byte vector 原样写盘所得的文件。用哪组 API 保存, 就用匹配的 API 恢复。

公共 get/set data 和文件包装在读写前同步 context; `get_size()` 是大小计算。同步只等待已提交工作, 不能把尚未执行的 pending shift/copy 自动变成已完成的 memory 更新。本周 checkpoint 选在正常 decode 已完成、没有悬而未决 memory 操作的稳定边界。

### 当前 state 没有保存什么

`llama_context::state_write_data()` 只写入 `llm_arch_name(model.arch)` 与 `memory->state_write(io)`; 对应 read 函数核对架构后读取 memory。它没有序列化 logits、embedding output buffer 或外部 sampler RNG。

头文件中 `llama_state_get_size()` 附近仍有提及 logits/embedding 的注释, 不能用该注释替代实际实现。因此“load 成功后直接 sample”并不是当前 API 普遍成立的恢复协议。

## 2. KV 序列化的实际内容

```text
context raw state
  -> architecture string
  -> memory state
       -> stream count
       -> 每个 stream:
            selected cell count
            selected cell metadata
            selected cells 的各层 K/V 数据及布局信息

context file
  -> session magic/version
  -> token count + caller-provided tokens
  -> context raw state

sequence raw bytes
  -> internal magic + source seq ID
  -> 指定 sequence 的 memory state

sequence file
  -> sequence magic/version
  -> token count + caller-provided tokens
  -> 指定 sequence 的 memory state
```

普通 KV 的 `state_write()` 先按目标 seq ID 选择有效 cells, 组织物理连续区间以读取 tensor。完整 memory 保存各 cell 的归属集合; 单 sequence 导出只保留该 sequence 的归属, 不把其他 owner 一并移植。

cell metadata 包含 position、sequence 数与 IDs, 必要时还包含扩展位置。数据部分包括 V 布局、层数、类型、行大小等信息以及 K/V 数值。当前普通路径没有把 token ID 历史、allocator head、物理空洞或 pending shift 队列作为应用 checkpoint 完整记录下来。

### 为什么恢复后的物理 cell 可以不同

- 完整 KV 恢复按保存顺序重建有效 cells, 不要求保持原来的空洞布局。
- 单 sequence 恢复将序列化的 owner 映射成 `dest_seq_id`, 用 `find_slot(ubatch,false)` 重新分配目标空槽, 再通过 `slot_info` 写入 tensor。
- 连续 slots 可以批量写, 非连续 slots 采用分散写入。逻辑 position 与 K/V 对应正确即可, 不以 cell index 与原来相等作为验收条件。

单 sequence 导出再恢复通常也不会自动恢复它原先与其他 sequence 的物理共享关系。共享关系、物理布局、语义等价和字节完全相同是不同验收目标。

## 3. 五段已有测试的真实断言

以下来自 [test-save-load-state.cpp](../../../../tests/test-save-load-state.cpp), 是代码阅读结果, 不是本次运行报告。

| 测试段 | 准备和动作 | 实际检查 | 不能据此推出 |
| --- | --- | --- | --- |
| Test 1 `test_baseline` | 新建 context 与 dist sampler; `common_prompt_batch_decode(...,save_state=true)` 在末 prompt token 之前保存; 再 decode 末 token, 生成参考序列 | prompt decode 成功, 参考 token vector 非空; 为后续测试提供结果 | Test 1 自己已经完成恢复验证; 该函数也没有逐项验证文件内容 |
| Test 2 `test_seq_rm_isolated` | 显式 `n_ctx=256,n_seq_max=2,kv_unified=true`; 分别向 seq 0 和 seq 1 prefill 相同前缀, 最多各 128 token; 保存 seq 1 bytes, 删除 seq 0, 再保存 seq 1 | state 大小非零、写入数等于分配长度、删除返回 true、`state_before == state_after` | 它通过 `seq_cp()` 建立了共享前缀; 实际没有调用 `seq_cp`, 也未比较独立续写基线 |
| Test 3 `test_state_load` | 新 context 加载 Test 1 文件; 令 `n_past=n_token_count_out-1`; decode 最后 prompt token; 用同 seed 新 sampler 生成 | load 成功、末 token decode 成功、生成结果非空且 token IDs 与参考 vector 相等 | state 文件恢复了可直接采样的 logits; 测试实际主动重放了未入 memory 的末 token |
| Test 4 `test_seq_cp_host` | 新建支持两条 sequence 的 context, 加载文件, 先 replay 末 token; 导出 seq 0 host state; clear(true); 恢复到 seq 1, 继续生成 | 导出与导入字节数分别等于 byte vector 长度, 结果 token IDs 等于参考 | 新 context 仅导入 sequence bytes 就可 sample; 第一次 sample 使用的是同一 context 在 clear 之前 replay 产生的 logits |
| Test 5 `test_seq_cp_device` | 与 Test 4 相同的时序, 但用 `*_ext(...,ON_DEVICE)` 保存和恢复 | 字节数一致, 续写 token IDs 等于参考 | 必须有 GPU, 或返回 bytes 是独立、可长期持久化的普通磁盘快照 |

Test 4/5 名称里的 seq copy 是通过 state 导出/导入迁移, 不是对 `llama_memory_seq_cp()` 的共享行为测试。`llama_memory_clear()` 作用于 memory, 没有顺带销毁此前的 logits, 所以这两个特定测试的第一步采样不能推广成一般恢复规则。

这些测试显式使用 `llama_sampler_init_dist(params.sampling.seed)`, 并在各对照阶段新建相同 seed 的 sampler。加命令行 `--temp 0` 不会改变这段显式 sampler 构造为 greedy。`generate_tokens()` 循环采样 `n_predict` 次, 每次记录 token ID 并 decode, 本身没有以 EOG 提前终止的分支; 它也没有逐 logits 比较。

还有两项容易漏掉的证据边界:

1. `common_prompt_batch_decode()` 只有 `save_state && n_new>1` 时走“保留末 token 未 decode”的保存分支, 该分支还要求 `n_new<=n_batch`。本周自建协议明确规定 `P>=2`。
2. 该 helper 调用 `llama_state_save_file()` 时未检查 bool 返回值; 后续 Test 3 加载和比较覆盖了部分结果, 但自建实验仍应直接检查保存结果。Test 3 中加载出的 token 数被使用, token 列表本身没有与原始列表逐项比较, 重放 token 来自原始 `tokens.back()`。

学习路线记录的历史测试通过情况不构成本次执行证据。本文没有重新运行这五段测试, 也没有生成新的 `dump_state.bin`。

## 4. checkpoint 时间线

本周采用“最后一个已知输入尚未 decode”的边界。令 `prompt=t[0]..t[P-1]`, `P>=2`。

```text
准备 checkpoint:
  decode t[0]..t[P-2], positions 0..P-2
  -> 完成计算并同步
  -> memory 的 pos_max = P-2
  -> 保存 memory bytes 和原始字节数
  -> 应用另存 pending_token=t[P-1], pending_pos=P-1

baseline:
  decode pending_token at pending_pos, output=true
  -> 本次 decode 的 fresh logits
  -> 新 greedy sampler 采样 g0
  -> decode g0 at P
  -> 采样 g1, 依相同规则继续

restore:
  clear 或创建匹配的新 context
  -> 用原始 bytes/长度恢复到合法 seq
  -> 核对 pos_max=P-2, 恢复应用字段
  -> decode 同一个 pending_token at P-1, output=true
  -> 这次 decode 的 fresh logits
  -> 新 greedy sampler 采样 g0'
  -> decode g0' at P, 依同一规则继续
  -> 比较 g0,g1,... 与 g0',g1',... 的 IDs 及停止原因
```

例如 P=4 时, 保存的是 p0/p1/p2 的状态, pending 为 t3@p3。恢复后第一次供采样使用的新 logits 来自 **`decode(t3,position=3,target_seq,output=true)`**。它不是 load 生成的, 也不是已被消费的末位置重复 decode 所得。

`common_prompt_batch_decode()` 保存文件时可以传入全部 P 个 token, 而当时 memory 只有 P-1 个 token。文件里的 token 数不是 memory 已消费 token 数, 也不天然等于下一输入 position。

如果采用“最后 token 已 decode”的另一种 checkpoint, 就需要另行定义输出保存或合法重放/回退协议, 不能无条件重放已存在的位置。当前协议通过保存 pending 输入避免这项歧义。

## 5. 文件内容与应用侧字段

| state / 文件实际提供 | 应用仍需保存或主动固定 |
| --- | --- |
| memory 的位置、sequence metadata 和 tensor 数据 | 完整 token 历史, 已消费到哪里, pending token/position, 当前 `n_past/next_pos` |
| context state 中的架构标识 | 精确模型文件标识或校验值, tokenizer/template 配置; 相同架构不代表相同权重 |
| 文件包装中的调用者 token 列表 | 该列表的语义, 它是否包含 pending 输入; 不由文件自动解释 |
| 单 sequence 状态及恢复目标参数 | 逻辑会话到目标 seq ID 的映射, 所有后续 batch 应使用的 seq ID |
| 部分类型、宽度、层数和布局校验信息 | build、backend、K/V 类型、Flash Attention、context/sequence 容量、RoPE 等运行参数 |
| 原始已保存 bytes | 保存时的实际长度, 使用的 API 格式与 flags, 文件路径或数据完整性记录 |
| 不包含外部 sampler RNG | sampler 配置、接受历史和必要状态; greedy 基线可新建 greedy sampler |
| 不包含输出 buffer 和应用循环状态 | 生成计数、上限、EOG/停止规则、采样输出行的映射 |

随机采样在生成中途已经推进 RNG 和接受历史, 仅把 seed 重设为初值不一定能续接。这里让 baseline 和 restored 都从尚未进行生成采样的相同边界使用 greedy, 便于验证 memory 恢复。

### 为什么加载成功不等于模型和参数完全匹配

context state 会检查架构字符串, KV read 还检查 stream 数、层数、容量、K/V 类型、行大小和 V 转置布局等。源码也明确保留了补充模型一致性信息的 TODO。没有完整权重内容校验, 相同架构、相同形状但权重不同仍不代表相同计算。

因此恢复实验主动固定同一模型、build、backend、F16 K/V、Flash Attention 设置与 context 参数。新 context 的迁移例子保持 `n_seq_max=2,kv_unified=true`, 将单 sequence host state 恢复到合法且为空的 seq 1, 并明确改用 seq 1 提交 pending 输入。只看 API 返回成功不足以证明续写一致。

### flags 的范围

- `PARTIAL_ONLY` 是由具体 memory 实现解释的部分状态请求。当前 `llama_memory_hybrid` 跳过 attention 部分、保留 recurrent 部分; 不能当完整 hybrid checkpoint。
- `ON_DEVICE` 将 tensor 数据保存在 backend buffer 中, host bytes 依赖 context 内的 `mem_storage`。它不以“必须是 GPU”为定义, 也不能脱离原数据存储当作独立磁盘文件。
- 同一 seq ID 后续再次获取 `ON_DEVICE` state 会使此前该类 device 数据失效。跨新 context 的持久化对照使用默认 host state。

## 6. 验收与跳过说明

- [x] API 表、五段测试的准备/动作/断言与未覆盖范围已完成。
- [x] 已定义 checkpoint 时间线, 列出应用字段与匹配条件。
- [x] 已明确恢复后用于第一次采样的 logits 来自 pending token 的新 decode。
- [ ] save/clear/restore、新 context 迁移、token IDs 对照和已有测试运行: 按要求暂时跳过, 不填写字节数、退出码或生成结果作为实测。

源码入口: [llama.h](../../../../include/llama.h) 的 state API 和 flags、[llama-context.cpp](../../../../src/llama-context.cpp) 的 `state_write_data/state_read_data/state_seq_*` 及公共包装、[llama-kv-cache.cpp](../../../../src/llama-kv-cache.cpp) 的 `state_write/state_read/state_read_meta/state_read_data`、[common.cpp](../../../../common/common.cpp) 的 `common_prompt_batch_decode`。
