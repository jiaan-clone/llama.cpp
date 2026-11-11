# week8 day1: CLI 调用链、依赖边界与运行环境

本文回答[第 8 周学习路线](../../008-llama-cpp-week8-daily-learning-route.md)中 Day 1 的当天输出与验收。

核对基准为当前 checkout `697b4bbf4633a38abfd66f7cc54a167095aedef3`, 与路线中的 `545ddb30a` 不同。已阅读源码并只读检查本机环境、已有构建缓存和文件。按要求跳过重新构建、运行 `--help`、CLI 推理及手动断点实验。路线要求的笔记与环境记录集中放在本文, 未创建 `results/week8/` 下的运行证据。

## 1. 当前 CLI 的完整调用图

```text
tools/cli/main.cpp
  -> llama_cli(argc, argv)
       -> common_init()
       -> common_params_parse(..., LLAMA_EXAMPLE_CLI)
       -> cli_context::init()
            -> 有 --server-base: 配置远端地址, 不在 CLI 中加载模型
            -> 无 --server-base: cli_server::start(params)
                 -> 选择可用端口, 复制 server_params
                 -> std::thread -> llama_server(server_params, 0, nullptr)
                      -> server_context::load_model()
                           -> server_context_impl::load_model()
                                -> common_init_from_params(params_base)
                                     -> common_init_result
                                          -> llama_model_load_from_file()
                                          -> 创建 context 等推理状态
                      -> HTTP listener/workers + ctx_server.start_loop()
            -> 等待 /health, 获取 /props
       -> cli_context::run()
            -> 维护 system/user/assistant messages
            -> cli_context::generate_completion()
                 -> cli_client::post_sse("/v1/chat/completions", messages)
                 -> HTTP route -> 模板 -> tokens -> task -> queue
                 -> slot -> shared batch -> llama_decode -> 每 slot 采样
                 -> response queue -> SSE
            -> 累积 reasoning_content/content, 显示并追加 assistant 消息
```

本地模式中, CLI 与 Server 在同一进程中, `cli_server::start()` 使用 `std::thread`, 并没有创建独立的 Server 子进程。CLI 通过 `http://127.0.0.1:<port>` 发请求, 不是自己在终端输入循环中直接调用 `llama_decode()`。连接外部 Server 时, 模型由外部服务管理。

`common_init()`、`common_params_parse()` 与 `common_init_from_params()` 是不同职责: 前两者负责公共初始化和参数解析; 实际模型加载在第三者创建的初始化对象中发生。不能看到函数名中的 init 就推断它一定读取了模型权重。

源码入口: [cli.cpp](../../../../tools/cli/cli.cpp) 的 `llama_cli()`、[cli-context.cpp](../../../../tools/cli/cli-context.cpp) 的 `init/run/generate_completion()`、[cli-server.h](../../../../tools/cli/cli-server.h) 的 `start/wait_ready/stop()`、[cli-client.cpp](../../../../tools/cli/cli-client.cpp) 的 `post_sse()`、[server-context.cpp](../../../../tools/server/server-context.cpp) 的 `load_model()`、[common.cpp](../../../../common/common.cpp) 的 `common_init_result/common_init_from_params()`。

## 2. common、libllama 与产品层边界

| 层 | 提供的能力 | 不由这一层决定的事情 | 最终项目如何复用 |
| --- | --- | --- | --- |
| `libllama`, CMake target `llama` | model/context、tokenization、batch、memory、decode、核心 sampler API | Planner/Worker/Reviewer 的业务角色和轮数 | 共享一个 model, 为角色建立独立 context |
| `llama-common` | 参数映射、chat template、token 辅助、common sampler、Schema 转 grammar、日志和 RAII 辅助 | 计划是否符合用户需求、何时重试或终止项目 | 复用模板和采样, 避免自行拼接通用聊天格式 |
| CLI 产品层 | 终端交互、消息列表、HTTP client、SSE 显示 | 服务端实际调度和后端算子执行 | 学习 messages 与响应处理, 最终项目不必启动 CLI |
| Server 产品层 | HTTP 协议、请求校验、task/slot/batch 调度、响应格式 | 最终项目的多角色业务状态机 | 学习并发、取消、观测设计, 不增加 Agent route |
| 项目应用层 | role history、协议校验、候选与反馈传递、预算、trace | 模型矩阵运算与 backend 分派 | 固定顺序编排, 最多两轮 Worker/Reviewer |

`llama-common` 中的 chat template 使用仓库自己的 [Jinja 引擎](../../../../common/jinja/README.md)。它不是保存应用聊天历史的数据库; history 仍由调用它的应用维护。

### CMake 依赖如何印证调用图

```text
llama-cli
  -> llama-cli-impl
       -> llama-server-impl
            -> server-context -> llama-common + mtmd
            -> llama-ui + cpp-httplib
       -> llama-common
llama-common -> llama
```

见 [CLI CMake](../../../../tools/cli/CMakeLists.txt)、[Server CMake](../../../../tools/server/CMakeLists.txt)、[tools CMake](../../../../tools/CMakeLists.txt) 和 [common CMake](../../../../common/CMakeLists.txt)。当前 tools 的 CMake 在 `LLAMA_BUILD_SERVER` 条件内添加 CLI、Server 和 UI 子目录, 因此不能关掉 Server 后仍按当前路径构建 CLI。最终项目直接链接 `llama` 与 `llama-common`, 不需要复制 CLI 或 Server 的全部依赖与调度逻辑。

## 3. 与 llama-simple、llama-completion 的区别

| 入口 | 模型与生成的位置 | 主要学习价值 |
| --- | --- | --- |
| [llama-simple](../../../../examples/simple/simple.cpp) | 示例进程直接调用公共 API, 自己执行 tokenize、decode、sample 循环 | 理解模型、context、batch 与 token 的最小关系 |
| [llama-completion](../../../../tools/completion/completion.cpp) | 通过 `common_init_from_params()` 初始化, 在本地执行生成循环 | 学习 common 参数、模板、采样和交互能力的组合 |
| 当前 `llama-cli` | 作为 HTTP client, 连接内嵌线程中的或外部的 Server | 学习产品入口、消息管理与流式显示 |

一次最小生成的核心仍是: 加载 model -> 创建 context -> tokenize -> prefill prompt -> 从正确输出位置采样 -> 将新 token 加入下一次 decode -> 检查 EOG/预算 -> 释放资源。CLI 把其中的推理工作交给 Server, 没有改变这些基本步骤。

## 4. 运行环境记录

以下为本次只读检查结果, 不是推理或性能测试报告。

| 项目 | 记录 |
| --- | --- |
| 核对日期 | 2026-11-11 |
| checkout | `697b4bbf4633a38abfd66f7cc54a167095aedef3` |
| 工作目录 | `/home/king/llama.cpp` |
| 系统 | Linux `6.6.87.2-microsoft-standard-WSL2`, `x86_64` |
| CPU 信息 | `Intel(R) Core(TM) Ultra 9 285H` |
| 系统报告的总内存 | `/proc/meminfo` 的 `MemTotal: 32866988 kB`; 不代表全部可用于模型 |
| 本地模型文件 | `deepseek-r1-1.5b-f16.gguf`, 存在, 文件大小 `3560416128` bytes |
| Debug 缓存 | `build-study-debug/CMakeCache.txt` 存在, `CMAKE_BUILD_TYPE=Debug` |
| Debug backend 开关 | `GGML_CUDA=OFF`, `GGML_BLAS=OFF` |
| Debug 功能开关 | COMMON、TESTS、EXAMPLES、TOOLS、SERVER 均为 ON |
| Debug UI 开关 | `LLAMA_BUILD_UI=ON`, 与本周路线中的 OFF 不同 |
| Debug 编译器缓存 | `CMAKE_CXX_COMPILER=/usr/bin/c++`; 本次未运行编译器版本查询 |
| Debug 可执行文件 | `llama-cli`、`llama-server`、`llama-parallel` 均存在; 未运行, 不能证明与当前源码同步 |
| Release | `build-study-release/CMakeCache.txt` 和上述三个 Release 可执行文件均不存在 |

模型架构和 shape 可参照[第 7 周 Day1 的 GGUF 记录](../week7/week7-day1.md): 该文件的架构为 Qwen2。文件大小不等于实际 RSS, 本次没有重新加载模型或测量内存。

后续补实验时, 按路线配置 Debug/Release, 明确 CPU、线程 4、相同模型与模板、`temp=0`、`seed=42`。现有 Debug 缓存的 UI 开关差异应记录或按路线重新配置; Release 尚需构建。不能把现有文件存在当作本周构建通过。

## 5. 为什么不能给每个 Agent 调一次完整初始化

`common_init_result` 是拥有资源的对象。其内部持有 model、context、samplers 等 RAII 对象, `model()` 和 `context()` 返回供借用的指针。每次完整调用 `common_init_from_params(params)` 都会尝试加载一个 model, 并建立相应 context 等状态, 它不是自动查找并共享已有 model 的会话工厂。

给三个角色分别完整初始化会带来三次模型初始化及相应资源开销。即使 mmap 或操作系统页缓存可能共享部分文件页, 也不能据此宣称只有一个 model owner 或所有 backend 权重都共享。

本项目的设计答案是:

```text
model_runtime 持有唯一初始化 owner
  -> common_init_from_params(params, true), 使用 model_only 路径
  -> 从 owner 借用同一个 llama_model *
  -> Planner / Worker / Reviewer 分别 llama_init_from_model(shared_model, cparams)
  -> 各自管理 context、history 和该次调用的 sampler
```

`model_only=true` 仍然加载模型, 只是跳过后续完整 context 等初始化; 它不是“不加载模型”。runtime 必须比所有借用 model 的对象活得久, 不得把临时初始化对象销毁后留下裸指针, 也不得再次单独 free 由 owner 持有的 model。

## 6. 当天验收与跳过项

| 路线要求 | 本文交付 | 状态 |
| --- | --- | --- |
| CLI 调用图和边界表 | 第 1-3 节 | 已完成源码分析 |
| 版本与运行环境 | 第 4 节, 含实际缓存差异 | 已完成只读核对 |
| 构建、三个帮助文件 | 沿用路线中的命令, 本次未执行 | 暂时跳过, 待补实际输出 |
| 一次短 CLI 请求 | 需保存模型/模板、stdout/stderr、reasoning/content、停止原因 | 暂时跳过, 没有实测结果 |
| 解释单次模型加载 | 第 5 节 | 已给出设计答案, 运行证明待补 |

短 CLI 请求后续应区分正常 EOG、token 上限和流内 error。reasoning 模型在 64 tokens 内没有最终 content, 不足以证明 CLI 调用链失败; 同样, 能显示部分 reasoning 也不代表完整回答已通过验收。
