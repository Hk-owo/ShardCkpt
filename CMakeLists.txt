cmake_minimum_required(VERSION 3.24)
project(ckpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(CKPT_BUILD_TESTS "构建 C++ 单元测试" ON)

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
