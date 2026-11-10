cmake_minimum_required(VERSION 3.24)
project(ckpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(CKPT_BUILD_TESTS "构建 C++ 单元测试" ON)
option(CKPT_BUILD_PYTHON "构建 Python 扩展 ckpt_native（需要 pybind11）" ON)

find_package(Threads REQUIRED)

file(GLOB_RECURSE CKPT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

find_path(LIBURING_INCLUDE_DIR liburing.h)
find_library(LIBURING_LIBRARY uring)
if(NOT LIBURING_INCLUDE_DIR OR NOT LIBURING_LIBRARY)
  message(FATAL_ERROR "找不到 liburing，安装方式：apt-get install -y liburing-dev")
endif()

add_library(ckpt_core STATIC ${CKPT_SOURCES})
target_include_directories(ckpt_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(ckpt_core PRIVATE ${LIBURING_INCLUDE_DIR})
target_link_libraries(ckpt_core PUBLIC Threads::Threads PRIVATE ${LIBURING_LIBRARY})

if(MSVC)
  target_compile_options(ckpt_core PRIVATE /W4)
else()
  target_compile_options(ckpt_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CKPT_BUILD_TESTS)
  enable_testing()
  add_executable(ckpt_local_file_test tests/cpp/local_file_test.cpp)
  target_link_libraries(ckpt_local_file_test PRIVATE ckpt_core)
  add_test(NAME ckpt_local_file_test COMMAND ckpt_local_file_test)
endif()

# --- Python 扩展 -------------------------------------------------------------
#
# ckpt_native 是 ckpt_core 的薄包装：只做异常映射、GIL 释放、buffer 零拷贝和
# 线程归属检查。构建期依赖 pybind11（不进 requirements.txt）：
#
#   uv pip install --python .venv/bin/python pybind11
#   cmake -S . -B build -G Ninja \
#         -DPython3_EXECUTABLE=.venv/bin/python \
#         -Dpybind11_DIR=$(uv run --python .venv/bin/python python -m pybind11 --cmakedir)
#
# 必须显式给 Python3_EXECUTABLE：否则 CMake 可能找到系统里的另一个 python（版本
# 不同会编出 ABI 对不上的 .so，导入时报 "compiled with ... but ... is required"）。
#
# 产物落在 build/python/ckpt_native*.so，native/__init__.py 负责把它挂上 sys.path。

if(CKPT_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG)
  if(NOT Python3_FOUND OR NOT pybind11_FOUND)
    message(FATAL_ERROR
      "构建 ckpt_native 需要 Python3 开发头文件与 pybind11。\n"
      "  uv pip install --python .venv/bin/python pybind11\n"
      "  cmake ... -DPython3_EXECUTABLE=.venv/bin/python \\\n"
      "        -Dpybind11_DIR=$(uv run --python .venv/bin/python python -m pybind11 --cmakedir)\n"
      "只想构建 C++ 部分可以加 -DCKPT_BUILD_PYTHON=OFF")
  endif()

  pybind11_add_module(ckpt_native MODULE bindings/python.cpp)
  target_link_libraries(ckpt_native PRIVATE ckpt_core)
  set_target_properties(ckpt_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
endif()
