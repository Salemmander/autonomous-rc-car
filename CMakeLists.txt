cmake_minimum_required(VERSION 3.16)
project(autonomous-rc-car LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(steer_test tests/steer_test.cpp)

target_include_directories(steer_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_options(steer_test PRIVATE -Wall -Wextra)
