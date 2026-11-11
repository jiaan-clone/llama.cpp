# week7 day2: Backend 对象、设备能力与实验环境

本文回答[第 7 周学习路线](../../007-llama-cpp-week7-daily-learning-route.md)中 Day 2 的当天输出与验收。

核对基准为当前 checkout `697b4bbf4633a38abfd66f7cc54a167095aedef3`。已阅读接口、注册和 CPU 实现, 并只读检查本地模型文件; 按要求跳过构建、硬件诊断、设备枚举、能力测试和短推理。下表中的 CPU 能力是源码定义, 不是本机二进制的探测结果。

## 1. Backend 对象关系图

```text
registry: ggml_backend_reg
  | enumerate devices
  v
device: ggml_backend_device
  | init_backend()                 | get_buffer_type()
  v                                v
backend: ggml_backend             buffer type: ggml_backend_buffer_type
  | graph_compute / copy / sync     | alloc_buffer(size)
  |                                v
  |                              buffer: ggml_backend_buffer
  |                                | actual or wrapped storage
  +---- reads/writes tensors ------>+---- tensor.data at an offset

scheduler
  -> holds backend list and compute buffer types
  -> assigns ops, splits graph, arranges copies and execution
  -> gallocr plans and allocates graph tensor storage
```

一个 registry 可以暴露多个 device, 一个 device 可以初始化多个 backend 执行实例; buffer type 也可以分配多个 buffer。device 并非只表示物理 GPU, 还包括 CPU 和 ACCEL 抽象。backend handle 表示执行实例, 不表示一块权重内存。

| 对象 | 主要责任 | 源码入口 |
| --- | --- | --- |
| registry | 实现注册、设备枚举、扩展函数查询 | `ggml_backend_reg_i` |
| device | 名称、类型、内存、能力查询, 初始化执行实例 | `ggml_backend_device_i` |
| backend | 图执行、异步传输接口、同步和 event 操作 | `ggml_backend_i` |
| buffer type | 分配方式、alignment、最大分配、tensor allocation size、host 可访问性 | `ggml_backend_buffer_type_i` |
| buffer | 存储实例、基址、tensor set/get/copy、用途和容量 | `ggml_backend_buffer_i` |
| scheduler | 按放置、能力和优先级协调执行 | `ggml_backend_sched_*` |

接口定义见 [ggml-backend.h](../../../../ggml/include/ggml-backend.h) 和 [ggml-backend-impl.h](../../../../ggml/src/ggml-backend-impl.h)。tensor metadata 通常在 host, `data` 却可能指向设备地址, 不能因此直接用 CPU 解引用。

## 2. 注册、发现与执行实例初始化

[ggml-backend-reg.cpp](../../../../ggml/src/ggml-backend-reg.cpp)既支持按编译宏注册 backend, 也提供动态加载路径。`register_backend()` 将 registry 及其 devices 加入列表; `ggml_backend_load_all()` 走可用实现的加载逻辑。编译出了库、加载了 registry、发现了 device、成功执行 op 是不同阶段。

[llama-context.cpp](../../../../src/llama-context.cpp)在非 vocab-only 路径中先为模型选定 devices 初始化 backend, 再加入 ACCEL backend, 最后加入 CPU backend。随后用 backend 列表及计算 buffer types 创建 scheduler。CPU 通常处于最后的回退位置, 前面的 backend 具有更高调度优先级。

`llama-bench --list-devices` 主要展示可用于 offload 的设备, 列表没有 CPU 不等于 CPU backend 缺失。真正的 CPU 验证要结合注册日志、CPU 初始化以及具体算子测试。仅有 `libggml-cuda.so` 文件也不能证明 CUDA 能运行。

## 3. 三种能力查询的区别

| 查询 | 回答的问题 | 不能推出什么 |
| --- | --- | --- |
| `supports_op(device, op)` | 这个具体 type、shape、layout 的 op 是否被支持 | 不能推出这个名字下所有配置都支持, 也不是数值测试通过 |
| `supports_buft(device, buft)` | 执行实例能否使用这种 buffer type 中的数据 | 不能推出它支持所有使用该 buffer 的运算 |
| `offload_op(device, op)` | 即使权重所在 buffer 不兼容, 是否倾向接手此运算 | 不能推出必然更快, 也不能免除 copy 和同步 |

例如 scheduler 对 CPU host 权重检查更高优先级 backend 的 `supports_op` 与 `offload_op`, 可能决定把计算搬过去; 目标不能直接访问源 buffer 时还要安排复制。CPU 的 `offload_op` 接口为 NULL, 通用包装函数将缺省情况返回 false。

## 4. 设备能力表

下表依据 [ggml-cpu.cpp](../../../../ggml/src/ggml-cpu/ggml-cpu.cpp) 的 `ggml_backend_cpu_device_get_props()` 和 `ggml_backend_cpu_device_supports_op()`。默认 buffer 指普通 CPU buffer, 不包含全部架构特定 extra buffer 的行为。

| device name | type | default buft | host buffer | async | events | MUL_MAT support | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CPU`, 实际型号待查 | CPU | `ggml_backend_cpu_buffer_type()` | 默认 buffer 在 host; 专用 `caps.host_buffer=false`, `buffer_from_host_ptr=true` | false | false | 普通 buffer 下, `src1` 为 F32 或 `src0` 对应的 `vec_dot_type`; 完整合法 shape 和具体路径仍需验证 | CPU 源码静态检查, 未运行测试 |
| CUDA device, 名称待查 | 待设备枚举 | 待查询 | 待查询 | 待查询 | 待查询 | 待对具体 tensor 查询并运行 | 本次跳过设备诊断 |

`caps.host_buffer=false` 指没有额外的专用 host buffer 接口, 不表示 CPU 不能访问系统内存。GPU 的 host buffer 常用于 pinned memory, 与默认 device buffer 不同。CPU `supports_buft()` 接受 host buffer 或其支持的 extra buffer; extra buffer 的 op 能力还会交给专用实现判断。

以普通合法连续布局的 `W=[256,16,1,1]`, `X=[256,1,1,1]` 为例, W 为 F16 或 Q4_0, X 为 F32 时满足通用 CPU MUL_MAT 的输入类型条件。这是源码推断, `support` 命令、reference 比较和实际线程记录均未执行。

## 5. 模型、构建、硬件与参数记录

本次已取得的只读事实:

| 项目 | 记录 |
| --- | --- |
| 源码 commit | `697b4bbf4633a38abfd66f7cc54a167095aedef3` |
| 模型路径 | `/home/king/llama.cpp/deepseek-r1-1.5b-f16.gguf` |
| 文件大小 | 3,560,416,128 bytes, 为文件长度, 不是 RSS 或完整驻留权重容量 |
| GGUF | version 3, 339 个 tensor, `general.architecture=qwen2`, `general.file_type=1` |
| 核心配置 | 28 层, hidden 1536, FFN 8960, Q heads 12, KV heads 2 |
| 工作区状态 | 路线文件已有本地修改, week7/week8 答案目录有未跟踪文件; 本次不替换已有路线修改 |

下面是补做实验时应保存的证据清单, 不是声称 `results/week7/` 中已经有这些文件:

| 应保存的文件 | 需要记录的内容 | 当前状态 |
| --- | --- | --- |
| `commit.txt`, `worktree-diff-stat.txt` | 实验时实际源码版本和未提交修改 | commit 已在本文记录, 独立日志未生成 |
| `model-size.txt`, 后续 `model-sha256.txt` | 模型大小和同源权重身份 | 大小已记录, hash 未计算 |
| `lscpu.txt`, `memory.txt` | CPU 型号、核/线程、NUMA、总内存 | 暂时跳过 |
| `cmake-debug.txt`, `cmake-release.txt` | 实际 CMakeCache, 编译器、CPU 指令选项及 backend 开关 | 暂时跳过 |
| `bench-help.txt`, `quantize-help.txt` | 实际所用二进制的参数帮助 | 暂时跳过 |
| `devices.txt` | 可用 offload devices 和加载情况 | 暂时跳过 |
| CUDA 诊断与短推理日志 | driver、Toolkit、编译器、初始化和执行结果 | 暂时跳过 |

计划配置沿用路线: Debug 用于断点, Release 用于性能; `GGML_CUDA=OFF`, `GGML_BLAS=OFF`, common/tests/examples/tools 开启, server/UI 关闭。还需检查 Vulkan、SYCL 等是否被已有 cache 打开。`WEEK7_THREADS=4` 是待验证的起点, 不等于本机最优值; CPU 指令集、线程绑定和 OpenMP 等配置要在两组量化对比中保持一致。

Debug 计划 target 为 `llama-simple`, `llama-bench`, `test-backend-ops`, `test-quantize-fns`; Release 还需要 `llama-quantize`, `llama-completion`, `llama-perplexity`。本次没有运行任何构建命令, 不能将这些计划参数写成已生效配置。

CUDA 分支本次选择跳过, 原因是用户要求暂不做手动实验。尚未确认本机是否存在可用 NVIDIA GPU, 也未重测 driver/runtime, 不能把路线中的历史不匹配记载当成当前失败原因。后续需分别保存 `nvidia-smi`, `nvcc --version`, CMake 选用工具链和短推理结果; `nvidia-smi` 的 CUDA 版本不能代替 Toolkit 版本。

## 6. 三个验收问题

**buffer 在哪里?** 查看 tensor 的 buffer 或 view 基底 buffer, 再查 buffer type、所属 device、host 可访问性及实际存储包装方式。它回答数据的存储位置和访问规则。

**哪个 backend 执行?** 查看 scheduler 的节点分配和 split backend。同一个 host buffer 可能被 CPU 或某些 ACCEL backend 使用, 所以存储位置不能唯一决定执行者。

**是否需要 copy?** 检查消费者 backend 能否使用源 buffer type、是否需要另一份输入及同步条件。不同 backend 之间若 buffer 兼容, 可以直接访问; 如果不兼容, scheduler 会准备目标副本。即使不发生物理复制, 也仍需保证生产者先完成写入。

已完成对象图、接口边界、源码能力表与证据清单。实际构建、设备能力验证及 `results/week7/` 原始日志均暂时跳过, 因而 Day 2 的环境实验验收尚未完成。
