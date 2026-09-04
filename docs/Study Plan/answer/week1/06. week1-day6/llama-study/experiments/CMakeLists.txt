cmake_minimum_required(VERSION 3.12) # 声明最低支持的 CMake 版本，低于3.12时，配置阶段直接失败
project(llama_simple_study LANGUAGES CXX) # 定义一个CMake 项目，项目名为 llama_simple_study, 只启用C++ 编译器，CMake 会在此处检测 c++、编译器能力等
# 定义 LLAMA_PREFIX 变量，表示 llama.cpp 的安装前缀目录。
# CACHE PATH ： 把变量存入构建目录的 CMakeCache.txt， 后续重新配置时仍保留。也可以在配置时覆盖它，例如：cmake -S . -B ../build -DLLAMA_PREFIX=/其他/安装目录
set(
    LLAMA_PREFIX
    "/home/king/llama-study/.llama-install"
    CACHE PATH "llama.cpp install prefix"
)

# 查找已经安装的 llama.cpp CMake 包
# Llama：包名；对应安装生成的 llama-config.cmake。
# CONFIG：只按“包提供的配置文件”寻找，不使用 CMake 内置的 FindLlama.cmake 模块模式。
# REQUIRED：找不到包时立刻配置失败。
# PATHS：指定寻找配置文件的目录。
# NO_DEFAULT_PATH：不去 /usr/local、系统路径等其他位置找，确保链接的是 LLAMA_PREFIX 指向的这份 llama.cpp。
# 该配置成功后，会提供可链接的 CMake target， 例如：llama 
find_package(Llama CONFIG REQUIRED 
    PATHS "${LLAMA_PREFIX}/lib/cmake/llama" #
    NO_DEFAULT_PATH
)

# 定义构建目标：
# 目标名：llama-simple-study
# 源文件：同目录的 llama-simple-study.cpp
# 构建产物：llama-simple-study 可执行文件
# 这只是声明依赖图；真正编译发生在：cmake --build /home/king/llama-study/build
add_executable(llama-simple-study llama-simple-study.cpp)

# 让可执行文件链接 llama 库。
# llama-simple-study：使用依赖的一方。
# llama ：find_package(Llama) 提供的导入 target。
# PRIVATE：llama 只供当前可执行文件链接；如果未来有其他 target 依赖 llama-simple-study，不会把 llama 继续传递出去。对可执行文件来说，这是合适的。
target_link_libraries(llama-simple-study PRIVATE llama)

# 要求该目标以至少 C++17 编译。CMake 会为当前编译器选择对应参数，例如 GCC 的 -std=gnu++17 或 -std=c++17。
target_compile_features(llama-simple-study PRIVATE cxx_std_17)

# 设置构建产物的运行时动态库搜索路径（RPATH）。这样编译出的程序运行时能自动找到动态库，不必每次手动写LD_LIBRARY_PATH=...
set_target_properties(llama-simple-study PROPERTIES 
    BUILD_RPATH "${LLAMA_PREFIX}/lib")
