# week7 day5: Block quantization、Q4_K_M 与 imatrix

本文回答[第 7 周学习路线](../../007-llama-cpp-week7-daily-learning-route.md)中 Day 5 的当天输出与验收。

源码基准为 `697b4bbf4633a38abfd66f7cc54a167095aedef3`。已只读核对本地 GGUF 的实际 tensor 名称、type 和 shape, 并按源码完成存储及类型选择推导。按要求跳过量化 dry-run、实际模型转换、imatrix 校准、推理和质量测试。`quantization-decisions.csv` 草案放在本文, 目标类型是静态预期, 不是 dry-run 输出。

## 1. Block 大小手算

定义见 [ggml-common.h](../../../../ggml/src/ggml-common.h), 当前 `QK_K=256`, `K_SCALE_SIZE=12`, `ggml_half` 为 2 bytes。取连续权重 `W.ne=[256,16,1,1]`, 共 `256*16=4096` 个值:

| type | 每 block 值数 | block 组成 | 每 block bytes | bits/value | 每行 bytes | tensor bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| F16 | 1 | 一个 half | 2 | 16 | `256*2=512` | `512*16=8192` |
| Q8_0 | 32 | FP16 d + 32 个 int8 | `2+32=34` | `34*8/32=8.5` | `8*34=272` | `272*16=4352` |
| Q4_0 | 32 | FP16 d + 16 个 packed bytes | `2+16=18` | `18*8/32=4.5` | `8*18=144` | `144*16=2304` |
| Q4_K | 256 | FP16 d/dmin + 12-byte scales/mins + 128-byte quants | `2+2+12+128=144` | `144*8/256=4.5` | 144 | `144*16=2304` |

这些数值不包含 GGUF metadata、文件 alignment、backend padding 或其他 tensor。Q4_0 与 Q4_K 的有效位数相同, 不表示 block 布局、误差或 dot 实现相同。

相应连续 `nb[4]` 为:

```text
F16:  [2,   512, 8192, 8192]
Q8_0: [34,  272, 4352, 4352]
Q4_0: [18,  144, 2304, 2304]
Q4_K: [144, 144, 2304, 2304]
```

对 block 类型, `nb[0]` 是一个 block 的存储大小, 不是每个逻辑值的独立字节步长。必须先沿 `nb[1]` 找到行, 再找 block 并解码其中的值。

## 2. Q4_0 量化与解量化示意

[ggml-quants.c](../../../../ggml/src/ggml-quants.c) 的 `quantize_row_q4_0_ref()` 对每 32 个值操作:

```text
32 个浮点值 x
  -> 找到绝对值最大的带符号数 vmax
  -> d = vmax / -8, id = d != 0 ? 1/d : 0
  -> d 保存为 FP16
  -> q[j] = min(15, int8(x[j]*id + 8.5))
  -> 每字节保存两个 nibble

重建:
  q = 从 nibble 取出的 0..15 编码
  x_hat = FP16_to_FP32(d) * (q - 8)
  error = x_hat - x
```

这里的 d 可以是负数, 不能把这个实现误写成恒正的 `absmax/7`。误差包括整数编码的舍入/范围限制和 scale 存入 F16 的误差; 这是有损压缩。

举例, 一个 32 值 block 的绝对值最大数为 -8, 其他值不超过该幅度, 则 `d=1`。其中 `x=1.3` 编码为 `int8(1.3+8.5)=9`, 重建为 `1*(9-8)=1`, 误差为 `-0.3`。这是说明算法的自选数据, 不是本地模型实际权重或测得的整体误差。

### 为什么不能用 float 指针读取一个量化值

对于某行中的第 k 个 Q4_0 值:

```text
block_id = k / 32
r = k % 32
j = r % 16
byte = block[block_id].qs[j]
q = r < 16 ? (byte & 0x0f) : (byte >> 4)
value = FP16_to_FP32(block[block_id].d) * (q - 8)
```

同一个 byte 的低 nibble 对应 block 中第 j 个值, 高 nibble 对应第 `j+16` 个值, 并非简单的相邻两个值。行由 `nb[1]` 定位, block 由 18-byte 大小定位; 存储里有 scale 和 packed integers, 将 `data` 强制解释为 `float *` 会读错格式。

Q8_0 的 reference 路径取 `d=absmax/127`, 保存 F16 scale, 再将 `x/d` 用 `roundf` 转成 int8; 重建近似为 `d*q`。Q4_K 在 256 值 super-block 内划分 8 组 32 值, 用 d/dmin 和打包的 scale/min 信息表达组内重建, 可概括为 `x_hat=d*scale*q-dmin*min`。它比 Q4_0 的单 scale 结构更复杂, 但这里不把全部参数优化过程等同于这条重建公式。

## 3. Q4_K、Q4_K_M 与 KV 类型分别是什么

| 名字 | 层级 | 含义 |
| --- | --- | --- |
| `GGML_TYPE_Q4_K` | 单 tensor 存储格式 | 按 256 值、144 bytes 的 block 编码 |
| `Q4_K_M` | 整个模型的量化预设 | 按 tensor 角色、层位置、架构和参数选用混合类型 |
| `-ctk/-ctv` | 推理 context 的 K/V cache 类型 | 决定缓存的 K/V 表示, 不由权重量化预设自动决定 |

Q4_K_M 的默认候选类型是 Q4_K, 但一些输出或敏感权重会用 Q6_K 等更高精度类型, norm/bias 等还可能保留原类型。Q6_K 当前每 block 为 `128+64+16+2=210` bytes, 256 个值, 有效 `6.5625` bits/value。因此不能用“全部参数乘 4 bits”估算整个 Q4_K_M GGUF。

CPU Q4_K 的通用 dot 路径还可能把 F32 activation 临时量化为 Q8_K, 这与 GGUF 中的权重类型、context KV 类型是三个不同对象。本周后续基线计划固定 `-ctk f16 -ctv f16`, 便于把权重量化的影响单独比较。

## 4. 从模型预设到逐 tensor 决策

调用链见 [quantize.cpp](../../../../tools/quantize/quantize.cpp) 和 [llama-quant.cpp](../../../../src/llama-quant.cpp):

```text
llama-quantize: 解析 preset / overrides / imatrix / dry-run
  -> llama_model_quantize()
  -> llama_model_quantize_impl()
       -> 读取模型 metadata 与 tensor 信息
       -> tensor_allows_quantization()
       -> llama_tensor_get_type()
            -> 显式 embedding/output/tensor override
            -> 非 pure 模式的 llama_tensor_get_type_impl()
            -> shape 不兼容时的 tensor_type_fallback()
       -> dry-run: 按目标 type 估算大小, 不转换权重
       -> 实际转换: 必要时将源数据转成 F32 工作数据
            -> llama_tensor_quantize_impl()
            -> ggml_quantize_chunk()
            -> 写入目标 GGUF
```

离线量化时使用 F32 工作数据, 与在线推理时把整个模型展开成常驻 F32 是两回事。后者不是 Day 4 通用量化 dot 的必经过程。

是否量化先看 tensor 是否符合条件: 普通一维 tensor 不走权重量化, norm、bias 及其他排除对象保留原类型; 二维 weight 再按预设挑选类型。`--pure`、手动 type overrides、保留 output 和 shape fallback 都会改变结果, 不能只看输出文件名判断实际分布。

## 5. 五个真实 tensor 的决策记录草案

来源为 `/home/king/llama.cpp/deepseek-r1-1.5b-f16.gguf`, 实际架构 Qwen2, 28 层, FFN 8960, vocab 151936。下面五个名称、F16 源类型和 GGUF shape 均已只读核对。

推导条件: Q4_K_M, 无 imatrix, `pure=false`, 不覆盖 tensor type, 不剪层, 不关闭 output 量化。模型具有独立 `output.weight`, embedding 不按 tied-output 特例处理。所有列出的 K 都满足 256 对齐, 因而预期无需 block 对齐 fallback。

| tensor | GGML shape `[K,M]` | 源 bytes | 预期目标 | 预期目标 bytes | 选择原因 |
| --- | --- | ---: | --- | ---: | --- |
| `token_embd.weight` | `[1536,151936]` | 466747392 | Q4_K | 131272704 | 非 tied embedding, 保持默认 Q4_K |
| `blk.0.attn_q.weight` | `[1536,1536]` | 4718592 | Q4_K | 1327104 | Q4_K_M 的 Q projection 未触发升精度分支 |
| `blk.0.ffn_up.weight` | `[1536,8960]` | 27525120 | Q4_K | 7741440 | 普通 FFN up 保持默认类型 |
| `blk.0.ffn_down.weight` | `[8960,1536]` | 27525120 | Q6_K | 11289600 | 首个 down 计数为 0, 满足 `use_more_bits(0,28)` |
| `output.weight` | `[1536,151936]` | 466747392 | Q6_K | 191439360 | 非 Falcon、维度对齐、无 override 的 output 提升精度分支 |

`use_more_bits(i,n)` 当前判断为 `i<n/8 || i>=7*n/8 || (i-n/8)%3==2`。本表只依赖首个 FFN down 的 `i=0`, 即 `0<28/8=3` 为真; 不把这一条推广成所有层都用 Q6_K。其他 tensor 还要结合实际遍历计数和角色判断。attention V 也有按计数升精度的规则, 不能用 Q 的结果代替 V。

例如 FFN down 的 Q6_K 大小为 `(8960/256)*1536*210=11289600`; output 大小为 `(1536/256)*151936*210=191439360`。这些只是连续 tensor payload 大小, 不是实际 buffer allocation 或整份量化文件大小。

以下是 `results/week7/quantization-decisions.csv` 的内嵌草案。`source_evidence` 表示源信息已读取, `decision_evidence` 单独表示目标尚未 dry-run 核对:

```csv
tensor_name,source_type,shape,source_bytes,preset,target_type_predicted,estimated_payload_bytes,reason,source_evidence,decision_evidence
token_embd.weight,F16,"[1536,151936]",466747392,Q4_K_M,Q4_K,131272704,untied_embedding_default,local_GGUF_descriptor,source_derivation_dry_run_pending
blk.0.attn_q.weight,F16,"[1536,1536]",4718592,Q4_K_M,Q4_K,1327104,attention_Q_default,local_GGUF_descriptor,source_derivation_dry_run_pending
blk.0.ffn_up.weight,F16,"[1536,8960]",27525120,Q4_K_M,Q4_K,7741440,FFN_up_default,local_GGUF_descriptor,source_derivation_dry_run_pending
blk.0.ffn_down.weight,F16,"[8960,1536]",27525120,Q4_K_M,Q6_K,11289600,use_more_bits_counter_0_of_28,local_GGUF_descriptor,source_derivation_dry_run_pending
output.weight,F16,"[1536,151936]",466747392,Q4_K_M,Q6_K,191439360,output_higher_precision,local_GGUF_descriptor,source_derivation_dry_run_pending
```

如果改用 Q8_0 预设, 在上述默认条件下这五个 tensor 预期均为 Q8_0, 依次为 `247959552,2506752,14622720,14622720,247959552` bytes。这同样只是用 `(K/32)*M*34` 算出的 payload 预期, 没有运行第二次 dry-run。

## 6. Imatrix 的用途与限制

[imatrix.cpp](../../../../tools/imatrix/imatrix.cpp) 的 `IMatrixCollector::collect_imatrix()` 通过图执行 callback 观察相关 matmul 的 activation `src1`。普通 MUL_MAT 当前要求 activation 为 F32 且 `ne[1]>=16`, 主要收集 `blk.` 权重, `output.weight` 需要启用 `process_output`; `MUL_MAT_ID` 有单独的 expert 统计路径。

普通 dense 路径按输入通道累加 `x[j]*x[j]` 并累计样本数量。加载 GGUF imatrix 时, quantize 工具按 counts 归一化统计值; 它近似描述校准数据上各输入通道的二阶幅度, 不是完整的通道协方差矩阵。

```text
calibration text -> tokenize -> model forward
  -> callback observes activations
  -> sum of squares + counts for each covered input channel
  -> imatrix file
  -> quantize loads and normalizes entries
  -> per-tensor importance data -> ggml_quantize_chunk(..., imatrix)
  -> weighted quantization decisions where the format uses them
```

直观上, 如果某输入通道经常出现较大的 activation, 同样的权重误差可能对输出产生更大影响。量化时可以利用这类信息加权误差, 可概括为关注 `sum_j importance[j]*(w[j]-w_hat[j])^2`, 但不能说所有格式都严格优化同一个完整目标, 或恢复了所有跨通道相关性。

imatrix 不是训练或微调模型, 不更新原始浮点权重来学习新任务。其作用受校准文本、token 数、context、tensor 覆盖及目标量化格式限制; callback 的筛选条件也意味着并非所有 tensor 都自动有统计。

Q4_K_M 可以不提供 imatrix。部分其他低位格式要求提供, 当前代码会按 tensor 的目标类型检查; dry-run 成功也可能伴随“实际转换需要 imatrix”的警告。应检查对应日志, 不能把所有量化格式概括成同一种要求。imatrix 信息在部分预设中还可能影响类型策略, 但不能据此声称本表的 Q4_K_M 目标必然改变。

选做时需记录校准语料来源、token 数、context、覆盖 tensor、命令参数和模型身份, 并使用独立评估集判断质量。用同一批文本校准再评估, 不能证明泛化改善。工具约定见 [imatrix README](../../../../tools/imatrix/README.md); 本次未运行校准。

## 7. 模型变小为什么不保证 PP/TG 按比例变快

量化主要减少权重的存储和潜在读取流量, 同时增加 block 解码、scale 运算、activation 转换等工作。它不会按同样比例减少 attention、KV 访问、norm、采样、线程同步或框架开销。

PP 的多个 token 可以复用权重, 浮点矩阵路径可能已经有效利用计算单元; 量化后的额外转换或不同 kernel 可能抵消读取收益。单 sequence TG 的权重读取占比可能较大, 因而更容易受益, 但长 context 的 KV/attention 开销、缓存、模型大小和线程条件可以改变结果。

文件大小也不等于每步实际 DRAM/PCIe 传输量, mmap 文件页是否驻留、专用 buffer 布局和跨设备 copy 都要单独看。速度和质量还是不同维度: 数值误差变大不自动说明任务质量不可用, 算子测试通过也不证明文本/JSON 质量与 F16 相同。

## 8. 当天验收状态与后续实验选择

- 已完成: 四种格式的 block/row/tensor 手算、Q4_0 编解码示意、Q4_K_M 与 KV 类型区分、五个真实 tensor 的静态决策表、imatrix 作用和限制、PP/TG 性能不能按文件压缩比推断的解释。
- 暂时跳过: Q4_K_M/Q8_0 dry-run、磁盘容量检查、量化模型生成、imatrix 校准、速度及质量测试。原因是本次要求跳过手动实验, 没有量化成功、性能提升或质量验收通过的结论。
- 后续默认选择: 资源允许时从同一份 F16 模型生成无 imatrix 的 Q4_K_M, 固定 CPU Release、线程、KV F16 与输入后完成对照。资源不足时保留 dry-run 和容量原因, 改做 F16 的单变量线程对照; 这种降级只能完成 CPU 性能部分, 不能声称完成了量化质量对照。

真正补做 dry-run 后, 用原始日志核对上述五行的目标类型和字节数, 将 `source_derivation_dry_run_pending` 替换成日志出处。dry-run 仅验证决策与估算, 仍不能代替生成模型后的数值、质量和性能实验。
