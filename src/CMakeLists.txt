# llama.cpp 核心库的 CMake 配置文件。
#
# 本文件负责定义名为 llama 的库目标：收集模型实现和核心推理源码，设置库的版本、
# ABI 及头文件搜索路径，要求使用 C++17，并将目标链接到 ggml。顶层 CMakeLists.txt
# 通过 add_subdirectory(src) 引入本文件；最终生成的 llama 库会被工具、示例和应用复用。

llama_add_compile_flags() # 应用项目统一的编译器警告、Sanitizer 等编译参数。

#
# 库目标配置。
#

# llama 核心库。

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp") # 自动收集 models/ 目录下的模型实现源文件。

add_library(llama # 定义 llama 库目标；库类型由 BUILD_SHARED_LIBS 全局选项决定。
            ../include/llama.h # 将对外公开的 C API 头文件纳入目标文件列表。
            llama.cpp # 核心 llama API 和推理流程实现。
            llama-adapter.cpp # LoRA 等模型适配器实现。
            llama-arch.cpp # 模型架构相关定义与处理逻辑。
            llama-batch.cpp # 批量推理输入数据结构和处理逻辑。
            llama-chat.cpp # 聊天模板及消息格式处理。
            llama-context.cpp # 推理上下文的创建、配置和生命周期管理。
            llama-cparams.cpp # 上下文参数的定义与校验。
            llama-grammar.cpp # 语法约束和结构化输出支持。
            llama-graph.cpp # 计算图构建和相关操作。
            llama-hparams.cpp # 模型超参数读取与管理。
            llama-impl.cpp # 核心 API 的内部实现辅助代码。
            llama-io.cpp # 模型及相关数据的输入输出功能。
            llama-kv-cache.cpp # Key/Value 注意力缓存的通用实现。
            llama-kv-cache-iswa.cpp # ISWA 模式的 KV 缓存实现。
            llama-kv-cache-dsa.cpp # DSA 模式的 KV 缓存实现。
            llama-kv-cache-msa.cpp # MSA 模式的 KV 缓存实现。
            llama-kv-cache-dsv4.cpp # DSV4 模式的 KV 缓存实现。
            llama-memory.cpp # 通用模型内存管理逻辑。
            llama-memory-hybrid.cpp # 混合内存模式实现。
            llama-memory-hybrid-iswa.cpp # 混合 ISWA 内存模式实现。
            llama-memory-recurrent.cpp # 循环模型的内存管理实现。
            llama-mmap.cpp # 使用内存映射加载模型文件。
            llama-model-loader.cpp # 模型文件加载器实现。
            llama-model-saver.cpp # 模型文件保存器实现。
            llama-model.cpp # 模型对象及其生命周期管理。
            llama-quant.cpp # 模型量化相关逻辑。
            llama-sampler.cpp # Token 采样策略实现。
            llama-vocab.cpp # 词表加载、查询和 Token 转换。
            unicode-data.cpp # Unicode 数据表及相关常量。
            unicode.cpp # Unicode 编码和字符处理逻辑。
            unicode.h # Unicode 处理接口头文件。
            ${LLAMA_MODELS_SOURCES} # 添加前面自动收集的各模型架构源文件。
            ) # 结束 llama 库源文件列表。

set_target_properties(llama PROPERTIES # 为 llama 目标设置库版本和平台相关属性。
    VERSION ${LLAMA_INSTALL_VERSION} # 设置完整库版本，用于生成带版本号的库文件。
    SOVERSION 0 # 设置 ABI 主版本号，当前固定为 0。
    MACHO_CURRENT_VERSION 0 # 保持 macOS Mach-O 当前版本较小，避免链接器处理过大的版本号。
) # 结束 llama 目标属性设置。

target_include_directories(llama PRIVATE .) # 仅 llama 内部代码可使用 src/ 目录下的头文件。
target_include_directories(llama PUBLIC ../include) # 将 include/ 目录作为公共头文件搜索路径传播给使用者。
target_compile_features   (llama PRIVATE cxx_std_17) # llama 目标使用 C++17，保持该标准版本不再提升。

target_link_libraries(llama PUBLIC ggml) # 链接 ggml，并将该依赖传递给链接 llama 的下游目标。

if (BUILD_SHARED_LIBS) # 构建共享库时启用额外的平台和导出设置。
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON) # 生成位置无关代码，满足共享库链接要求。
    target_compile_definitions(llama PRIVATE LLAMA_BUILD) # 仅在构建 llama 本身时定义导出控制宏。
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED) # 向使用者声明当前链接的是共享版 llama 库。
endif() # 结束共享库配置判断。
