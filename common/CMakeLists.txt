# llama.cpp 公共组件的 CMake 配置文件。
#
# 本文件负责构建两个公共目标：llama-common-base 用于编译并提供自动生成的构建信息，
# llama-common 则汇集命令行参数、聊天模板、采样、JSON/语法、HTTP、缓存、Jinja 等通用功能。
# 同时，它配置线程库、可选的子进程和 LLGuidance 支持，并声明这些目标与核心 llama 库之间的依赖。

find_package(Threads REQUIRED) # 查找并要求系统提供线程库，以便后续链接 Threads::Threads。

llama_add_compile_flags() # 应用项目统一的编译器警告、Sanitizer 等编译参数。

#
# llama-common-base：构建信息基础库。
#

# 构建信息源文件配置。

if(EXISTS "${PROJECT_SOURCE_DIR}/.git") # 只有在项目根目录存在 Git 元数据时才尝试读取提交信息。
    set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git") # 记录项目 .git 目录或 gitfile 的路径。

    # 判断当前项目是否作为 Git 子模块存在。
    if(NOT IS_DIRECTORY "${GIT_DIR}") # 子模块的 .git 通常是指向真实 Git 目录的文本文件。
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK) # 读取 gitfile 中保存的真实 Git 目录路径。
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK}) # 提取 gitdir: 后面的路径。
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS) # 判断提取出的路径是否以斜杠开头，即是否为绝对路径。
        if (SLASH_POS EQUAL 0) # 绝对路径可以直接作为 Git 目录使用。
            set(GIT_DIR "${REAL_GIT_DIR}") # 保存绝对形式的真实 Git 目录。
        else() # 相对路径需要相对于项目根目录解析。
            set(GIT_DIR "${PROJECT_SOURCE_DIR}/${REAL_GIT_DIR}") # 拼接得到真实 Git 目录的绝对路径。
        endif() # 结束 Git 目录路径类型判断。
    endif() # 结束 Git 子模块路径解析。

    if(EXISTS "${GIT_DIR}/index") # Git 索引存在时可以用它触发构建信息重新生成。
        # 为下面生成的 build-info.cpp 配置自动重新运行 CMake 的依赖。
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIT_DIR}/index") # Git 提交或暂存区变化时重新配置项目。
    else() # 找不到索引文件时仍可继续构建，但无法可靠地追踪 Git 变化。
        message(WARNING "Git index not found in git repository.") # 提醒用户 Git 索引缺失。
    endif() # 结束 Git 索引检查。
else() # 项目不是 Git 仓库或 .git 元数据不存在。
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.") # 提醒用户无法自动生成完整构建信息。
endif() # 结束 Git 仓库检查。

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in") # 指定构建信息 C++ 源文件模板。
set(OUTPUT_FILE   "${CMAKE_CURRENT_BINARY_DIR}/build-info.cpp") # 指定配置后生成的 C++ 源文件路径。

configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE}) # 用当前构建变量替换模板并生成 build-info.cpp。

set(TARGET llama-common-base) # 将后续操作的目标名称设置为 llama-common-base。
add_library(${TARGET} STATIC ${OUTPUT_FILE}) # 使用生成的源文件构建静态基础库。

target_include_directories(${TARGET} PUBLIC .) # 将 common/ 目录作为该基础库及其使用者的头文件搜索路径。

if (BUILD_SHARED_LIBS) # 当主项目构建共享库时，基础静态库也必须能参与共享库链接。
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON) # 为基础库生成位置无关代码。
endif() # 结束基础库 PIC 配置判断。

#
# llama-common：公共工具库。
#

set(TARGET llama-common) # 将后续操作的目标名称切换为 llama-common。

add_library(${TARGET} # 定义 llama-common 公共工具库目标。
    arg.cpp # 命令行参数解析实现。
    arg.h # 命令行参数解析接口。
    base64.hpp # Base64 编码和解码工具。
    chat-auto-parser-generator.cpp # 自动生成聊天模板解析器的实现。
    chat-auto-parser-helpers.cpp # 聊天模板自动解析的辅助逻辑。
    chat-auto-parser.h # 聊天模板自动解析器接口。
    chat-diff-analyzer.cpp # 聊天消息差异分析实现。
    chat-peg-parser.cpp # 基于 PEG 的聊天模板解析实现。
    chat-peg-parser.h # PEG 聊天模板解析器接口。
    chat.cpp # 聊天消息和模板处理实现。
    chat.h # 聊天处理公共接口。
    common.cpp # 公共工具函数实现。
    common.h # 公共工具函数接口。
    console.cpp # 控制台输入输出及终端交互实现。
    console.h # 控制台功能接口。
    debug.cpp # 调试辅助功能实现。
    debug.h # 调试功能接口。
    download.cpp # 文件下载功能实现。
    download.h # 文件下载接口。
    fit.cpp # 模型拟合或参数调优相关实现。
    fit.h # 拟合功能接口。
    hf-cache.cpp # Hugging Face 缓存访问实现。
    hf-cache.h # Hugging Face 缓存接口。
    http.h # HTTP 客户端相关接口声明。
    imatrix-loader.cpp # I-matrix 数据加载实现。
    imatrix-loader.h # I-matrix 加载接口。
    json-schema-to-grammar.cpp # 将 JSON Schema 转换为语法约束。
    llguidance.cpp # LLGuidance 结构化生成集成代码。
    log.cpp # 日志输出实现。
    log.h # 日志接口和日志级别声明。
    ngram-cache.cpp # N-gram 缓存实现。
    ngram-cache.h # N-gram 缓存接口。
    ngram-map.cpp # N-gram 映射实现。
    ngram-map.h # N-gram 映射接口。
    ngram-mod.cpp # N-gram 模型相关实现。
    ngram-mod.h # N-gram 模块接口。
    peg-parser.cpp # 通用 PEG 语法解析实现。
    peg-parser.h # PEG 解析器接口。
    preset.cpp # 预设配置加载和处理实现。
    preset.h # 预设配置接口。
    reasoning-budget.cpp # 推理预算处理实现。
    reasoning-budget.h # 推理预算接口。
    sampling.cpp # 采样参数及采样辅助逻辑。
    sampling.h # 采样功能接口。
    speculative.cpp # 推测解码相关实现。
    speculative.h # 推测解码接口。
    subproc.cpp # 子进程创建和管理实现。
    subproc.h # 子进程功能接口。
    trie.cpp # Trie 前缀树实现。
    trie.h # Trie 前缀树接口。
    unicode.cpp # Unicode 字符处理实现。
    unicode.h # Unicode 字符处理接口。
    jinja/lexer.cpp # Jinja 模板词法分析器实现。
    jinja/lexer.h # Jinja 词法分析器接口。
    jinja/parser.cpp # Jinja 模板语法解析器实现。
    jinja/parser.h # Jinja 解析器接口。
    jinja/runtime.cpp # Jinja 模板运行时实现。
    jinja/runtime.h # Jinja 运行时接口。
    jinja/value.cpp # Jinja 模板值类型实现。
    jinja/value.h # Jinja 值类型接口。
    jinja/string.cpp # Jinja 字符串处理实现。
    jinja/string.h # Jinja 字符串处理接口。
    jinja/caps.cpp # Jinja 能力或特性检测实现。
    jinja/caps.h # Jinja 能力检测接口。
    ) # 结束 llama-common 源文件列表。

set_target_properties(${TARGET} PROPERTIES # 为 llama-common 设置库版本和平台相关属性。
    VERSION ${LLAMA_INSTALL_VERSION} # 设置完整库版本，用于生成带版本号的库文件。
    SOVERSION 0 # 设置 ABI 主版本号，当前固定为 0。
    MACHO_CURRENT_VERSION 0 # 保持 macOS Mach-O 当前版本较小，避免链接器处理过大的版本号。
) # 结束 llama-common 目标属性设置。

target_include_directories(${TARGET} PUBLIC . ../vendor) # 向目标及其使用者公开 common/ 和 vendor/ 头文件目录。
target_compile_features   (${TARGET} PUBLIC cxx_std_17) # 要求 llama-common 及其使用者使用 C++17。

if (LLAMA_SUBPROCESS) # 根据选项决定是否编译子进程功能。
    target_compile_definitions(${TARGET} PUBLIC LLAMA_SUBPROCESS) # 向目标及其使用者定义子进程功能宏。
endif() # 结束子进程功能判断。

if (BUILD_SHARED_LIBS) # 构建共享库时配置 llama-common 的位置无关代码和符号导出。
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON) # 生成可链接进共享库的位置无关代码。

    # TODO：未来改为更精细的符号导出控制。
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON) # Windows 下自动导出所有可导出的符号。
endif() # 结束共享库配置判断。

target_link_libraries(${TARGET} PUBLIC  llama-common-base) # 链接构建信息基础库，并将依赖传递给使用者。
target_link_libraries(${TARGET} PRIVATE cpp-httplib) # 私有链接 cpp-httplib，仅公共库内部使用其 HTTP 实现。

if (LLAMA_LLGUIDANCE) # 启用 LLGuidance 时，配置其外部 Rust 工程并链接生成的静态库。
    include(ExternalProject) # 引入用于管理外部项目下载和构建的 CMake 模块。
    set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source) # 指定 LLGuidance 源码目录。
    set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release) # 指定 Cargo release 构建产物目录。
    set(LLGUIDANCE_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}") # 按当前平台规则计算静态库文件名。

    ExternalProject_Add(llguidance_ext # 声明名为 llguidance_ext 的外部项目构建任务。
        GIT_REPOSITORY https://github.com/guidance-ai/llguidance # 指定 LLGuidance Git 仓库地址。
        # 对应 LLGuidance v1.0.1 的固定提交。
        GIT_TAG d795912fedc7d393de740177ea9ea761e7905774 # 固定源码提交，保证构建可复现。
        PREFIX ${CMAKE_BINARY_DIR}/llguidance # 指定外部项目的下载、临时和构建前缀目录。
        SOURCE_DIR ${LLGUIDANCE_SRC} # 指定外部项目源码目录。
        BUILD_IN_SOURCE TRUE # 在源码目录内直接执行构建命令。
        CONFIGURE_COMMAND "" # LLGuidance 不需要单独的 CMake 配置步骤。
        BUILD_COMMAND cargo build --release --package llguidance # 使用 Cargo 以 release 模式构建 llguidance crate。
        INSTALL_COMMAND "" # 不执行独立安装步骤，直接使用构建产物。
        BUILD_BYPRODUCTS ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME} ${LLGUIDANCE_PATH}/llguidance.h # 声明后续目标依赖的静态库和头文件产物。
        UPDATE_COMMAND "" # 禁止构建时自动更新 Git 仓库，保持固定提交。
    ) # 结束 LLGuidance 外部工程定义。
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_LLGUIDANCE) # 向目标及其使用者声明已启用 LLGuidance。

    add_library(llguidance STATIC IMPORTED) # 声明一个由外部工程生成的导入静态库目标。
    set_target_properties(llguidance PROPERTIES IMPORTED_LOCATION ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME}) # 指定导入库文件的位置。
    add_dependencies(llguidance llguidance_ext) # 确保使用导入库前先完成 LLGuidance 外部构建。

    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH}) # 让 llama-common 能找到外部生成的 llguidance.h。
    target_link_libraries(${TARGET} PRIVATE llguidance) # 将 LLGuidance 静态库私有链接到 llama-common。
    if (WIN32) # LLGuidance 在 Windows 上还需要以下系统库。
        target_link_libraries(${TARGET} PRIVATE ws2_32 userenv ntdll bcrypt) # 链接 Winsock、用户环境、NT 和加密 API 库。
    endif() # 结束 Windows 系统库链接判断。
endif() # 结束 LLGuidance 配置判断。

target_link_libraries(${TARGET} PUBLIC llama Threads::Threads) # 链接核心 llama 库和线程库，并向下游传递这两个依赖。
