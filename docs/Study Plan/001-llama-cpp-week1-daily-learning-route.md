# llama.cpp 第 1 周每日学习路线

本路线根据 [llama.cpp 8 周深度学习与二次开发计划](llama-cpp-8-week-learning-plan.md) 的“第 1 周: Transformer 最小模型、项目全貌和最小推理程序”拆分而来。目标不是读完很多文件, 而是用一个可计算的最小 Transformer 贯通下面这条链路:

```text
token id -> embedding -> decoder block -> hidden state -> logits
         -> sampler -> next token -> llama_decode(one token)
```

完成本路线后, 应能不看源码说明一次生成的主要 shape, 画出项目 target 依赖, 独立运行一个最小 CPU 推理程序, 并解释 `model`、`context`、`batch` 和 `sampler` 的生命周期。

## 使用约定

- 工作日每天 2 小时, 周末每天 4 小时, 总计约 18 小时。
- 统一使用 `B=1,T=3,D=4,H=2,Dh=2,V=8` 做纸面例子; `Dh=D/H`。逻辑 shape 采用 `[B,T,D]`, 看到 GGML `ne[]` 时以实际维度顺序为准。
- Transformer 术语先参考 [Transformer 概念桥接](000-transformer-concept-bridge.md), 再映射到源码。每个源码笔记都写出“对象含义、输入 shape、输出 shape、持久状态”。
- CPU 是本周硬性基线。以下命令默认在仓库根目录执行, 且已设置本地模型:

  ```bash
  cd /home/king/llama.cpp
  export MODEL="$PWD/Llama-3.2-3B-Instruct-f16.gguf"
  ```

- 学习笔记和实验代码放在仓外的 `llama-study/` 中, 不直接改动核心实现。建议目录:

  ```text
  llama-study/
    notes/week1-day{1..7}.md
    experiments/llama-simple-study.cpp
    results/week1/
  ```

## 每日固定节奏

每次开始先用 10 分钟闭卷回忆前一天的 3 个概念, 写下今天要验证的问题; 结束前留 15 分钟记录已确认事实、仍不确定的假设和下一次断点。中间时间用于源码阅读、纸面推导和可重复实验。周末的 4 小时可以拆成两个 2 小时 session, 中间休息 15 至 30 分钟。

## Day 1: 从 token 到 logits 的最小数值模型

**投入: 2 小时。** 今天只建立数据流和 shape, 不读复杂 C++。

### 学习路线

1. **0-10 分钟, 预热**: 写出 `token`、`token id`、`embedding`、`hidden state`、`logits` 的一句话定义, 不看 [概念桥接](000-transformer-concept-bridge.md)。
2. **10-35 分钟, 读概念**: 阅读概念桥接第 2 至 4 节, 确认 tokenizer 输出 `[B,T]`, embedding table 为 `[V,D]`, 第一层输入为 `[B,T,D]`。
3. **35-75 分钟, 做小例子**: 使用 `V=8,D=4,T=3` 画 embedding lookup。任选 3 个 token id, 给每个 id 写一个 4 维向量; 用表格标出 lookup 前后 shape, 不需要模拟真实权重。
4. **75-105 分钟, 接到输出**: 假设 block 是恒等函数, 画 `hidden [1,3,4] -> output projection [1,3,8] -> last-position logits [8]`。标出生成时为什么只取最后一个位置？
5. **105-120 分钟, 闭环**: 不看笔记口述“字符串如何变成 8 个候选 logits”, 并列出 3 个尚未理解的问题。

### 当天输出与验收

- `llama-study/notes/week1-day1.md`: 一张 token/embedding/hidden/logits 数据流图和 shape 表。
- 验收: 能解释 `V` 与 `D` 的区别; 能从 `[B,T]` 推到 `[B,T,D]` 再推到 `[B,T,V]`; 能说明为何采样通常使用最后一个位置的 logits。
- 未通过时: 回到概念桥接第 3、4 节, 重新用 2 x 3 和 3 x 2 小矩阵复习乘法和转置, 不进入 Day 2。

## Day 2: 一层 decoder block 的 shape 和残差

**投入: 2 小时。** 今天把 Day 1 的恒等 block 换成 Llama 风格的 pre-norm block。

### 学习路线

1. **0-10 分钟**: 闭卷写出 Day 1 的 4 个主要 shape。
2. **10-35 分钟**: 阅读 [概念桥接第 5 至 8 节](000-transformer-concept-bridge.md), 重点是 `RMSNorm`、residual、线性投影和 Q/K/V 的逻辑维度。
3. **35-70 分钟**: 画并标注下列结构, 每条箭头写 shape:

   ```text
   x [B,T,D]
     -> RMSNorm -> Attention [B,T,D] -> residual add -> x1 [B,T,D]
     -> RMSNorm -> FFN [B,T,D]       -> residual add -> x2 [B,T,D]
   ```

4. **70-105 分钟**: 为每个子层补充输入/输出和不变量: RMSNorm 的权重 `[D]`; Q/K/V 投影; attention 合并回 `[B,T,D]`; FFN 中间维度 `I` 大于 `D`, 最后压回 `D`。
5. **105-120 分钟**: 闭卷回答“为什么 residual 两侧 shape 必须相同?”和“为什么 attention/FFN 前后仍保持 `[B,T,D]`?”。

### 当天输出与验收

- `llama-study/notes/week1-day2.md`: 带 shape 的单层 decoder block 图, 至少标出两次 norm、两次 residual 和 FFN 中间维度。
- 验收: 能用 `B/T/D/H/Dh` 标出一层 block 的输入输出; 能用自己的话解释 RMSNorm 是尺度控制而不是 tokenizer; 能指出 residual add 不是 bias add。

## Day 3: Attention、RoPE、GQA 与 KV cache

**投入: 2 小时。** 今天完成最小 attention 计算并理解生成阶段的持久状态。

### 学习路线

1. **0-10 分钟**: 闭卷重画 Day 2 block, 标出 Attention 的输入和输出。
2. **10-30 分钟**: 阅读 [概念桥接第 8 至 12 节](000-transformer-concept-bridge.md), 先确认 Q/K/V 的角色, 再看 `QK^T -> scale -> causal mask -> softmax -> weighted V`。
3. **30-75 分钟, 手算**: 用 2 个 token、2 个 head、`Dh=2` 的整数向量手算一次简化 attention。至少写出 `Q`、`K`、`V`、`QK^T`、scale、mask、softmax 和加权 `V`; 未来位置的 score 必须被屏蔽。
4. **75-100 分钟, 位置和 head**: 在图上标出 RoPE 作用于 Q/K 的位置; 写一句话说明 GQA 中 `H` 个 Q heads 复用 `KvH` 个 K/V heads, 以及这如何减少 KV cache。
5. **100-120 分钟, 生成对比**: 做一张 prefill/decode 表: prompt 的多个 token 可以一起算, 新 token 通常一次追加一个; KV cache 保存历史 K/V, 避免每步重算历史。

### 当天输出与验收

- `llama-study/notes/week1-day3.md`: attention 手算过程、RoPE/GQA/KV cache 说明和 prefill/decode 对比表。
- 验收: 能解释当前位置为什么不能看未来 token; 能指出 RoPE 影响 Q/K 而不是直接生成 token; 能解释 `KvH < H` 时 cache 的 head 维度如何变化。
- 统一问题记录: 若 softmax 或矩阵维度算不通, 在笔记中保留错误版本和修正原因, 不用跳过计算。

## Day 4: 从公共 API 跑通最小推理

**投入: 2 小时。** 今天把纸面数据流映射到 `llama-simple` 的公共 C API。

### 学习路线

1. **0-10 分钟**: 复述 Day 3 的 Q/K/V、KV cache 和 prefill/decode 区别。
2. **10-45 分钟, 源码阅读**: 阅读 [`examples/simple/simple.cpp`](../examples/simple/simple.cpp), 按以下顺序在 Day 1 图上标注调用:

   ```text
   ggml_backend_load_all
   -> llama_model_load_from_file
   -> llama_tokenize
   -> llama_init_from_model
   -> llama_batch_get_one(prompt)
   -> llama_decode
   -> llama_sampler_sample
   -> llama_batch_get_one(next token)
   -> llama_decode(one token)
   ```

3. **45-70 分钟, API 对照**: 在 [`include/llama.h`](../include/llama.h) 中定位 model/context/batch/sampler 的默认参数和释放函数。记录 `n_ctx`、`n_batch` 的含义; 暂不深入内部 graph。
4. **70-100 分钟, 运行**: 若已有 Debug 构建, 执行:

   ```bash
   ./build-study-debug/bin/llama-simple \
     -m "$MODEL" -ngl 0 -n 8 \
     "Explain KV cache in one sentence."
   ```

   保存 stdout/stderr、commit id、模型路径和运行参数到 `llama-study/results/week1/day4/`。
5. **100-120 分钟, 复盘**: 在输出中区分 prompt 回显、生成 token、结束 token 和性能统计; 写下从 `llama_sampler_sample` 回到下一次 `llama_decode` 的循环条件。

### 当天输出与验收

- `llama-study/notes/week1-day4.md`: 公共 API 调用链图和每个句柄的创建/释放表。
- 一份可重复的 CPU 运行记录。
- 验收: 能指出 tokenize、prompt decode、sample、single-token decode 四个阶段; 能解释 `model` 为什么在 `context` 前创建, 以及 sampler 为什么最后释放。

## Day 5: 项目全貌、CMake target 与对象生命周期

**投入: 2 小时。** 今天建立项目边界, 不展开模型 loader 或 graph builder。

### 学习路线

1. **0-10 分钟**: 闭卷写出 Day 4 的 API 调用链和释放顺序。
2. **10-45 分钟, 目录导览**: 阅读 [`CMakeLists.txt`](../CMakeLists.txt)、[`src/CMakeLists.txt`](../src/CMakeLists.txt)、[`common/CMakeLists.txt`](../common/CMakeLists.txt)、[`app/CMakeLists.txt`](../app/CMakeLists.txt)。将应用层、`llama-common`、`libllama`、GGML、backend 的职责写成 5 行说明。
3. **45-70 分钟, target 验证**: 在已有构建目录执行:

   ```bash
   cmake --build build-study-debug --target help
   cmake --build build-study-debug --target llama-simple -j
   ```

   从输出确认 `llama-simple`、`llama`、`ggml` 和 CPU backend 的 target 名称; 不凭目录名猜依赖。
4. **70-100 分钟, 画图**: 画 `llama-simple -> llama -> ggml -> CPU backend` 的 target 依赖图, 旁边标注 `common/` 为可选共享工具层, `examples/simple` 直接使用 `libllama` 公共 API。
5. **100-120 分钟, 生命周期**: 完成 `model -> context -> batch -> sampler` 表, 分别写“创建者、使用者、释放函数、可否被多个 context 共享”。

### 当天输出与验收

- `llama-study/notes/week1-day5.md`: 目录/target 依赖图和 5 个核心对象生命周期表。
- 验收: 能说明 CMake 的 target 依赖而不是只背目录; 能回答 `model` 与 `context` 分离如何支持多个会话; 能解释 `n_batch` 的限制不等于 `n_ctx` 的容量。

## Day 6: 仓外复写最小程序并用 GDB 观察两阶段 decode

**投入: 4 小时。** 今天是本周的工程实践日, 先复写再调试, 不添加 sampling 策略。

### 学习路线

1. **0-20 分钟, 复盘和准备**: 检查 Debug 构建和 `$MODEL`; 建立 `llama-study/experiments/`、`results/week1/day6/`, 记录 `git rev-parse --short HEAD`。
2. **20-110 分钟, 仓外复写**: 复制 `examples/simple/simple.cpp` 到 `llama-study/experiments/llama-simple-study.cpp`, 删除参数解析和 encoder 分支, 固定 prompt 和 `n_predict=4`。保留 model/context/sampler 的显式创建和释放; 通过现有构建 target 或仓外 CMake 运行, 不修改 `src/`。
3. **110-150 分钟, 行为对照**: 用相同模型、prompt、`-ngl 0` 和 token 数运行原版与复写版, 比较 token 数、结束条件和性能输出。若采样器使用 greedy, 固定 seed 不是主要变量, 仍要记录完整命令。
4. **150-210 分钟, GDB 断点**: 运行:

   ```bash
   gdb --args ./build-study-debug/bin/llama-simple \
     -m "$MODEL" -ngl 0 -n 4 "Hello"
   ```

   在 GDB 中设置 `llama_model_load_from_file`、`llama_init_from_model`、`llama_context::decode`、`llama_sampler_sample` 断点。每次停住记录函数、prompt/decode 阶段、`batch.n_tokens` (或当前 batch 对应的 token 数)、`n_ctx/n_batch` 和返回值。
5. **210-240 分钟, 整理证据**: 画一条时间线, 明确第一次 decode 处理 prompt, 后续 decode 每次处理 1 个 token; 将 GDB 输出和原版/复写版对照结果写入 Day 6 笔记。

### 当天输出与验收

- `llama-study/experiments/llama-simple-study.cpp`。
- `llama-study/notes/week1-day6.md` 和 `llama-study/results/week1/day6/` 下的命令、日志、GDB 记录。
- 验收: 仓外程序可以完成至少 4 个生成 token 或遇到 EOG 正常退出; 能用断点证实 prompt 与 decode 的 batch token 数差异; 所有成功创建的对象都有对应释放。
- GDB 符号因构建选项不可见时, 先用 `set breakpoint pending on` 和 `info functions llama_` 确认符号, 不要为了断点修改核心源码。

## Day 7: 闭卷重建、端到端验收与补缺

**投入: 4 小时。** 今天不引入新 API, 只验证能否独立解释和重建第 1 周主线。

### 学习路线

1. **0-30 分钟, 闭卷画图**: 从 `token id` 开始, 画到 `next token id`; 在 Attention、FFN、logits、sampler 和 decode 循环处标出输入输出 shape。
2. **30-75 分钟, 口述（可录音）**: 不看源码完成一次 5 分钟解释, 必须覆盖 embedding、RMSNorm、Q/K/V、RoPE、causal mask、GQA、KV cache、residual、FFN 和 logits。把卡住的位置标为待补问题。
3. **75-135 分钟, 空白终端重建**: 从已配置的 build 目录开始, 重新运行 `llama-simple`、复写程序和一个 `ctest --test-dir build-study-debug -N`; 不复制 Day 4/6 的命令, 只允许查错。
4. **135-180 分钟, 端到端核对**: 对照 Day 1 shape 表、Day 5 架构图、Day 6 GDB 记录, 检查概念、API 和运行观察是否一致。特别核对 `n_ctx`、`n_batch`、`n_ubatch` 的边界, 本周只要求能解释前两者并知道 `n_ubatch` 将于后续周深入。
5. **180-220 分钟, 补缺**: 对每个未通过项回到对应文档章节, 最多补 40 分钟。若仍无法标出 Q/K/V、attention score、FFN 或 logits shape, 延长第 1 周而不要进入第 2 周。
6. **220-240 分钟, 周复盘**: 填写下方验收清单, 汇总本周事实、假设、命令和下一周的 3 个问题。

### 第 1 周验收清单

- [ ] 能用 `B/T/D/H/Dh/V` 标注 token、embedding、Attention、FFN 和 logits 的主要 shape。
- [ ] 能手算 2 个 token、2 个 head 的简化 attention, 并解释 causal mask。
- [ ] 能说明 RoPE、GQA、KV cache 分别解决什么问题。
- [ ] 能画出 `应用层 -> libllama -> GGML -> CPU backend` 的依赖关系。
- [ ] 能从 `llama_model_load_from_file()` 讲到 `llama_decode()`、`llama_sampler_sample()` 和下一次单 token decode。
- [ ] 能解释 `model`、`context`、`batch`、`sampler` 的所有权和释放顺序。
- [ ] 能在 CPU 上运行原版 `llama-simple` 和仓外复写程序, 并保留可复查日志。
- [ ] 能通过 GDB 记录 prompt 与逐 token decode 的 batch token 数差异。

## 本周完成包

在进入第 2 周前, `llama-study/` 中至少应有:

```text
notes/week1-day1.md       # token/embedding/logits shape
notes/week1-day2.md       # decoder block 图
notes/week1-day3.md       # attention 手算、RoPE/GQA/KV cache
notes/week1-day4.md       # 公共 API 调用链
notes/week1-day5.md       # 架构和 target 依赖
notes/week1-day6.md       # 复写和 GDB 记录
notes/week1-day7.md       # 闭卷验收和周复盘
experiments/llama-simple-study.cpp
results/week1/day4/       # 原版最小生成日志
results/week1/day6/       # 原版/复写版对照和 GDB 日志
```

如果机器没有可用的本地模型, 可以先完成 Day 1-3 和 Day 5 的纸面/源码任务; Day 4、Day 6、Day 7 的运行验收必须在有模型后补做。不要用 CUDA 驱动问题替代 CPU 验收, 第 1 周不要求 GPU。
