# llama.cpp 项目的顶层 CMake 构建脚本。
#
# 本文件负责统一配置项目的构建环境：声明项目与最低 CMake 版本，设置默认构建类型、
# 编译器参数和跨平台选项，管理 ggml 等第三方依赖，并按开关加入核心库、工具、示例、
# 测试及应用程序等子目录。文件末尾还定义库、公共头文件以及 CMake/pkg-config 包的安装规则，
# 使 llama.cpp 既可以独立构建，也可以作为其他 CMake 项目的子目录或可重定位软件包使用。

cmake_minimum_required(VERSION 3.14...3.28) # 要求 CMake 版本在 3.14 到 3.28 之间，以使用链接选项和隐式目标目录等特性。
project("llama.cpp" C CXX) # 声明项目名称，并启用 C 与 C++ 两种语言。
include(CheckIncludeFileCXX) # 引入用于检测 C++ 头文件是否存在的标准模块。

#set(CMAKE_WARN_DEPRECATED YES) # 可选：开启 CMake 弃用特性警告（当前保持关闭）。
set(CMAKE_WARN_UNUSED_CLI YES) # 对命令行中未使用的 -D 变量发出警告。

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json，便于 IDE 和静态分析工具使用。

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE) # 非 Xcode、非 MSVC 且未指定构建类型时设置默认值。
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE) # 将默认构建类型设为 Release，并写入缓存。
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo") # 为构建类型缓存提供常用选项。
endif() # 结束默认构建类型判断。

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") # 输出最终采用的构建类型，便于配置时确认。

# 添加自定义 CMake 模块搜索路径。
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/") # 将项目自带的 cmake/ 目录追加到模块搜索路径。

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # 将可执行文件输出到构建目录下的 bin/。
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # 将库文件输出到同一个 bin/ 目录。

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) # 判断当前项目是否是顶层（独立）CMake 工程。
    set(LLAMA_STANDALONE ON) # 顶层构建时启用独立构建标志。

    include(git-vars) # 读取 Git 版本等项目元数据。

    # 配置项目版本（当前逻辑预留）。
    # TODO # 待补充更完整的项目版本配置逻辑。
else() # 被其他 CMake 工程作为子目录引入时走此分支。
    set(LLAMA_STANDALONE OFF) # 子项目模式下关闭独立构建标志。
endif() # 结束顶层/子项目判断。

option(LLAMA_USE_SYSTEM_GGML "Use system libggml" OFF) # 是否使用系统已安装的 ggml 库，默认关闭。

option(LLAMA_WASM_MEM64 "llama: use 64-bit memory in WASM builds" ON) # WASM 构建是否启用 64 位线性内存，默认开启。

if (EMSCRIPTEN) # 检测是否正在使用 Emscripten 编译 WebAssembly。
    set(BUILD_SHARED_LIBS_DEFAULT OFF) # WASM 默认不生成共享库。

    # 使用 64 位内存，以支持 backend_get_memory 查询。
    # TODO：分析其性能影响，参考 https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using
    if (LLAMA_WASM_MEM64) # 根据开关决定是否使用 64 位 WASM 内存，以支持内存查询。
      add_compile_options("-sMEMORY64=1") # 为编译阶段传递 Emscripten 的 64 位内存选项。
      add_link_options("-sMEMORY64=1") # 为链接阶段传递相同的 64 位内存选项。
    endif() # 结束 WASM 64 位内存判断。
    add_link_options("-sALLOW_MEMORY_GROWTH=1") # 允许 WASM 运行时按需扩展线性内存。

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" OFF) # 是否将 WASM 嵌入生成的 llama.js，默认关闭。
    option(LLAMA_BUILD_HTML "llama: build HTML file" ON) # 是否额外生成可直接运行的 HTML 文件，默认开启。
    if (LLAMA_BUILD_HTML) # HTML 构建开启时调整可执行文件后缀。
        set(CMAKE_EXECUTABLE_SUFFIX ".html") # 将 Emscripten 产物命名为 .html。
    endif() # 结束 HTML 构建判断。
else() # 非 WASM 平台根据编译器决定共享库默认值。
    if (MINGW) # MinGW 环境下默认生成静态库，避免运行时依赖问题。
        set(BUILD_SHARED_LIBS_DEFAULT OFF) # 关闭共享库默认值。
    else() # 其他原生平台使用共享库默认值。
        set(BUILD_SHARED_LIBS_DEFAULT ON) # 开启共享库默认值。
    endif() # 结束 MinGW 判断。
endif() # 结束平台类型判断。

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT}) # 提供是否构建共享库的统一开关。

if (WIN32) # Windows 平台启用专用编译定义。
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS) # 禁用 MSVC 不安全 C 运行库函数警告。
endif() # 结束 Windows 平台判断。

if (MSVC) # MSVC 编译器使用以下专用编译参数。
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>") # 让 C 源文件按 UTF-8 编码编译。
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>") # 让 C++ 源文件按 UTF-8 编码编译。
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>") # 增大 C 编译目标可容纳的节数量。
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>") # 增大 C++ 编译目标可容纳的节数量。
endif() # 结束 MSVC 编译器判断。

if (LLAMA_STANDALONE) # 仅在独立构建时配置 Visual Studio 的并行构建行为。
    # enable parallel builds for msbuild
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true) # 允许 MSBuild 使用多工具任务并行编译。
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true) # 让不同构建共享进程数限制。
endif() # 结束独立构建判断。

if (CMAKE_SYSTEM_NAME STREQUAL "iOS") # iOS 平台通常不安装命令行工具。
    set(LLAMA_TOOLS_INSTALL_DEFAULT OFF) # 将工具安装开关默认设为关闭。
else() # 其他平台沿用是否独立构建作为默认值。
    set(LLAMA_TOOLS_INSTALL_DEFAULT ${LLAMA_STANDALONE}) # 独立构建默认安装工具，子项目默认不安装。
endif() # 结束工具安装默认值判断。

# subprocess spawning isn't a supported/sandbox-friendly operation on mobile OSes or in WASM
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID # 移动系统和 WASM 环境通常不支持安全地创建子进程。
        OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR EMSCRIPTEN) # 补充判断 Emscripten 平台标志。
    set(LLAMA_SUBPROCESS_DEFAULT OFF) # 在受限平台默认关闭子进程支持。
else() # 其他平台允许使用子进程。
    set(LLAMA_SUBPROCESS_DEFAULT ON) # 在常规桌面/服务器平台默认开启子进程支持。
endif() # 结束子进程默认值判断。

#
# 选项列表：下面集中声明用户可配置的构建开关。
#

# 调试相关选项。
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON) # 是否为项目代码启用全部编译器警告，默认开启。
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF) # 是否为第三方库启用全部警告，默认关闭以减少噪声。

# 构建相关选项。
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF) # 是否将警告视为错误（-Werror），默认关闭。

# Sanitizer（运行时检查器）相关选项。
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF) # 是否启用线程检查器，默认关闭。
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF) # 是否启用地址检查器，默认关闭。
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF) # 是否启用未定义行为检查器，默认关闭。

# 公共工具库相关选项。
option(LLAMA_BUILD_COMMON "llama: build common utils library" ${LLAMA_STANDALONE}) # 是否构建公共工具库；独立构建时默认开启。

# 测试、工具、示例等附加产物相关选项。
option(LLAMA_BUILD_TESTS     "llama: build tests"                                                                ${LLAMA_STANDALONE}) # 是否构建测试目标；独立构建时默认开启。
option(LLAMA_BUILD_TOOLS     "llama: build tools"                                                                ${LLAMA_STANDALONE}) # 是否构建命令行工具；独立构建时默认开启。
option(LLAMA_BUILD_EXAMPLES  "llama: build examples"                                                             ${LLAMA_STANDALONE}) # 是否构建示例程序；独立构建时默认开启。
option(LLAMA_BUILD_SERVER    "llama: build server example"                                                       ${LLAMA_STANDALONE}) # 是否构建服务器示例；独立构建时默认开启。
option(LLAMA_BUILD_APP       "llama: build the unified binary"                                                   ${LLAMA_STANDALONE}) # 是否构建统一可执行程序；独立构建时默认开启。
option(LLAMA_BUILD_UI        "llama: build the embedded Web UI for server"                                       ON) # 是否构建服务器内嵌 Web UI，默认开启。
option(LLAMA_USE_PREBUILT_UI "llama: use prebuilt UI from HF Bucket when available (requires LLAMA_BUILD_UI=ON)" ON) # 有可用预构建 UI 时是否优先使用，默认开启。

option(LLAMA_TOOLS_INSTALL "llama: install tools" ${LLAMA_TOOLS_INSTALL_DEFAULT}) # 是否在安装阶段安装工具目标。
option(LLAMA_TESTS_INSTALL "llama: install tests" ON) # 是否在安装阶段安装测试目标，默认开启。

# 第三方库相关选项。
option(LLAMA_OPENSSL    "llama: use openssl to support HTTPS" ON) # 是否使用 OpenSSL 提供 HTTPS 支持，默认开启。
option(LLAMA_SUBPROCESS "llama-common: use subprocess, required by server tools and server router mode" ${LLAMA_SUBPROCESS_DEFAULT}) # 是否让公共库启用子进程功能，默认值取决于平台。
option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF) # 是否将 LLGuidance 集成到公共工具库以支持结构化输出。


# 引入可重定位 CMake 包所需的配置。
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake) # 引入构建编号、提交哈希等版本信息定义。
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake) # 引入项目通用的 CMake 辅助函数和设置。

if (NOT DEFINED LLAMA_BUILD_NUMBER) # 如果用户未预先指定 llama 构建编号，则采用通用构建编号。
    set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER}) # 保存由 build-info.cmake 提供的构建编号。
endif() # 结束构建编号默认值判断。
if (NOT DEFINED LLAMA_BUILD_COMMIT) # 如果用户未预先指定提交标识，则采用通用提交标识。
    set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT}) # 保存由 build-info.cmake 提供的 Git 提交标识。
endif() # 结束提交标识默认值判断。
set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER}) # 根据构建编号生成用于安装包的语义化版本号。

# 覆盖 ggml 的相关选项，使其与 llama 保持一致。
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS}) # 让 ggml 使用与 llama 相同的全警告开关。
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS}) # 让 ggml 使用与 llama 相同的警告即错误开关。

# 调整以下 ggml 选项的默认值（仅在用户未指定时生效）。
if (NOT DEFINED GGML_LLAMAFILE) # 仅在用户未设置时调整 ggml 的 llamafile 默认选项。
    set(GGML_LLAMAFILE_DEFAULT ON) # 默认启用 llamafile 支持。
endif() # 结束 llamafile 默认值判断。

if (NOT DEFINED GGML_CUDA_GRAPHS) # 仅在用户未设置时调整 CUDA graphs 默认选项。
    set(GGML_CUDA_GRAPHS_DEFAULT ON) # 默认启用 CUDA Graphs 支持。
endif() # 结束 CUDA Graphs 默认值判断。

# 兼容旧选项的迁移辅助函数。
function (llama_option_depr TYPE OLD) # 定义兼容旧选项的辅助函数，参数为消息级别、旧选项和可选的新选项。
    if (${OLD}) # 仅当旧选项被启用时执行迁移提示。
        set(NEW "${ARGV2}") # 读取第三个参数作为新选项名称。
        if(NEW) # 如果存在对应的新选项，则执行迁移。
            message(${TYPE} "${OLD} is deprecated, use ${NEW} instead") # 按指定级别提示旧选项已弃用。
            set(${NEW} ON PARENT_SCOPE) # 在父作用域中启用新选项。
        else() # 没有替代选项时仅提示该选项将被忽略。
            message(${TYPE} "${OLD} is deprecated and will be ignored") # 输出弃用且忽略的提示。
        endif() # 结束新选项是否存在的判断。
    endif() # 结束旧选项是否启用的判断。
endfunction() # 结束兼容旧选项的函数定义。

llama_option_depr(FATAL_ERROR LLAMA_CUBLAS              GGML_CUDA) # 将旧的 LLAMA_CUBLAS 迁移到 GGML_CUDA，启用时直接报错。
llama_option_depr(WARNING     LLAMA_CUDA                GGML_CUDA) # 将旧的 LLAMA_CUDA 迁移到 GGML_CUDA 并给出警告。
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL) # 将旧的 LLAMA_METAL 迁移到 GGML_METAL 并给出警告。
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY) # 迁移旧的 Metal 嵌入库选项。
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE) # 将旧的 LLAMA_NATIVE 迁移到 GGML_NATIVE。
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC) # 将旧的 LLAMA_RPC 迁移到 GGML_RPC。
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL) # 将旧的 LLAMA_SYCL 迁移到 GGML_SYCL。
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16) # 将旧的 LLAMA_SYCL_F16 迁移到 GGML_SYCL_F16。
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN) # 将旧的 LLAMA_CANN 迁移到 GGML_CANN。
llama_option_depr(WARNING     LLAMA_CURL) # 提示旧的 LLAMA_CURL 选项已弃用且没有替代项。

# 引入许可证处理逻辑并登记本项目许可证文件。
include("cmake/license.cmake") # 加载项目许可证安装辅助函数。
license_add_file("llama.cpp" "LICENSE") # 将根目录 LICENSE 文件关联到 llama.cpp 组件。

#
# 第三方依赖配置。
#

if (LLAMA_USE_SYSTEM_GGML) # 用户要求使用系统中已安装的 ggml 时走此分支。
    message(STATUS "Using system-provided libggml, skipping ggml build") # 输出跳过内置 ggml 构建的状态信息。
    find_package(ggml REQUIRED) # 查找并要求系统提供的 ggml CMake 包。
    add_library(ggml ALIAS ggml::ggml) # 创建统一的 ggml 别名，兼容项目内部目标名称。
endif() # 结束系统 ggml 判断。

if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML) # 未找到 ggml 目标且未选择系统库时构建仓库内的 ggml。
    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER}) # 将 llama 的构建编号传递给 ggml 子项目。
    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT}) # 将 llama 的提交标识传递给 ggml 子项目。
    add_subdirectory(ggml) # 加入仓库内 ggml 子目录并构建其目标。
    # 如果仍未添加目标，则假定 ggml 已由父级 CMakeLists.txt 加入。
endif() # 结束内置 ggml 构建判断。

#
# 构建核心 llama 库。
#

add_subdirectory(src) # 加入 src 子目录，定义 llama 核心库目标。

#
# 构建公共库、程序、示例和测试。
#

if (LLAMA_BUILD_COMMON) # 启用公共工具库时构建 common 及其 HTTP 依赖。
    add_subdirectory(common) # 加入 common 子目录并构建 llama-common 库。
    add_subdirectory(vendor/cpp-httplib) # 加入 vendored 的 cpp-httplib 第三方库。
endif() # 结束公共工具库构建判断。

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION) # 公共库和测试均开启且非 CMake.js 构建时启用测试。
    include(CTest) # 引入 CTest 集成，注册测试相关选项和命令。
    add_subdirectory(tests) # 加入 tests 子目录并定义测试目标。
endif() # 结束测试构建判断。

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_EXAMPLES) # 公共库和示例开关同时开启时构建示例与概念验证程序。
    add_subdirectory(examples) # 加入 examples 子目录。
    add_subdirectory(pocs) # 加入 pocs 子目录。
endif() # 结束示例构建判断。

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS) # 公共库和工具开关同时开启时构建命令行工具集合。
    add_subdirectory(tools) # 加入 tools 子目录。
endif() # 结束工具构建判断。

if (LLAMA_BUILD_APP) # 启用统一应用程序开关时构建 app 子目录。
    add_subdirectory(app) # 加入 app 子目录并定义统一二进制目标。
endif() # 结束统一应用程序构建判断。

# 独立构建 libmtmd，不必引入完整的 tools/ 目录树。
# 适用于只为语言绑定打包 mtmd 库（例如 Apple XCFramework 或 WASM 构建）。
# 启用完整工具构建时，mtmd 已由上面的 tools/ 子目录构建；
# 只有在 LLAMA_BUILD_TOOLS 关闭时才执行此处，避免重复添加目标。
option(LLAMA_BUILD_MTMD "llama: build tools/mtmd library standalone" OFF) # 是否独立构建 tools/mtmd 库，默认关闭。
if (LLAMA_BUILD_MTMD AND NOT (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)) # 仅在未由完整 tools 构建 mtmd 时添加该子目录。
    add_subdirectory(tools/mtmd) # 单独加入 mtmd 子目录，供语言绑定或特定打包场景使用。
endif() # 结束独立 mtmd 构建判断。

#
# 安装和 CMake 包导出配置。
#

include(GNUInstallDirs) # 引入跨平台标准安装目录变量（如 bin、lib、include）。
include(CMakePackageConfigHelpers) # 引入生成可重定位 CMake 包配置文件的辅助函数。

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files") # 设置头文件安装目录，并允许用户通过缓存覆盖。
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files") # 设置库文件安装目录，并允许用户通过缓存覆盖。
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files") # 设置可执行文件安装目录，并允许用户通过缓存覆盖。

set(LLAMA_PUBLIC_HEADERS # 开始定义需要随公共库一起安装的头文件列表。
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h # 添加主要 llama C API 头文件。
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama-cpp.h) # 添加 llama C++ API 头文件并结束列表。

set_target_properties(llama # 为 llama 核心库设置安装相关属性。
    PROPERTIES # 开始指定目标属性键值对。
        PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}") # 声明公共头文件列表，供 install(TARGETS) 使用。

install(TARGETS llama LIBRARY PUBLIC_HEADER) # 安装 llama 库及其公共头文件。

if (LLAMA_BUILD_COMMON) # 仅当构建了公共库时才安装 llama-common。
    install(TARGETS llama-common LIBRARY) # 安装公共工具库目标。
endif() # 结束公共库安装判断。

configure_package_config_file( # 根据模板生成可重定位的 llama-config.cmake。
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama-config.cmake.in # 指定包配置模板文件。
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake # 指定生成文件的构建目录路径。
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama # 指定包配置文件的安装目录。
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR # 将头文件安装目录纳入可重定位路径处理。
              LLAMA_LIB_INSTALL_DIR # 将库文件安装目录纳入可重定位路径处理。
              LLAMA_BIN_INSTALL_DIR ) # 将二进制安装目录纳入可重定位路径处理并结束命令。

write_basic_package_version_file( # 生成供 find_package 使用的版本兼容文件。
        ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake # 指定版本文件输出路径。
    VERSION ${LLAMA_INSTALL_VERSION} # 写入前面计算出的安装版本号。
    COMPATIBILITY SameMajorVersion) # 允许同一主版本号内的版本被视为兼容。

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake # 安装生成的 CMake 包配置文件。
              ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake # 同时安装包版本描述文件。
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama) # 将两个 CMake 包文件放入标准包目录。

configure_file(cmake/llama.pc.in # 根据 pkg-config 模板生成 llama.pc 文件。
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc" # 指定生成的 pkg-config 文件路径。
        @ONLY) # 仅替换 @VAR@ 形式的变量，避免误替换其他内容。

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc" # 安装生成的 pkg-config 元数据文件。
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig) # 将其放入 pkg-config 的标准搜索目录。
