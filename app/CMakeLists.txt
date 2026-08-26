# llama.cpp 统一应用程序的 CMake 配置文件。
#
# 本文件定义一个内部目标 llama-app，并将其输出文件命名为 llama。该可执行程序复用
# server、CLI、补全、基准测试、量化和困惑度评估等实现库；同时收集 licenses/ 目录下
# 的附加许可证，生成许可证信息，并在启用工具安装时将最终程序加入安装规则。

set(TARGET llama-app) # 保存统一应用程序目标名称，供后续 CMake 命令复用。

add_executable(${TARGET} llama.cpp download.cpp) # 使用统一入口和下载功能源文件创建可执行目标。
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama) # 将生成的可执行文件命名为 llama，而不是内部目标名 llama-app。

target_link_libraries(${TARGET} PRIVATE # 开始声明统一程序需要链接的内部实现库。
    llama-server-impl # 服务器功能实现。
    llama-cli-impl # 交互式命令行界面实现。
    llama-completion-impl # 文本补全功能实现。
    llama-bench-impl # 性能基准测试实现。
    llama-batched-bench-impl # 批量推理基准测试实现。
    llama-fit-params-impl # 参数拟合功能实现。
    llama-quantize-impl # 模型量化功能实现。
    llama-perplexity-impl # 困惑度评估功能实现。
) # 结束内部实现库链接列表。
target_compile_features(${TARGET} PRIVATE cxx_std_17) # 要求统一应用程序使用 C++17 编译。

# 自动收集 licenses/ 目录中的附加许可证文件。
file(GLOB EXTRA_LICENSES "${CMAKE_SOURCE_DIR}/licenses/LICENSE-*") # 查找名称以 LICENSE- 开头的许可证文件。

foreach(FILE_PATH ${EXTRA_LICENSES}) # 遍历所有找到的附加许可证文件。
    get_filename_component(FILE_NAME "${FILE_PATH}" NAME) # 从完整路径中提取许可证文件名。
    string(REGEX REPLACE "^LICENSE-" "" NAME "${FILE_NAME}") # 去掉 LICENSE- 前缀，得到许可证名称。
    license_add_file("${NAME}" "${FILE_PATH}") # 将许可证名称和文件路径登记到项目许可证清单。
endforeach() # 结束附加许可证文件遍历。

license_generate(${TARGET}) # 为 llama-app 生成并嵌入或关联最终许可证信息。

if(LLAMA_TOOLS_INSTALL) # 仅在启用工具安装选项时安装统一应用程序。
    install(TARGETS ${TARGET} RUNTIME) # 将生成的 llama 可执行文件安装到标准运行文件目录。
endif() # 结束统一应用程序安装判断。
