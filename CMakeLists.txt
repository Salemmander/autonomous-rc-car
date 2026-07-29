cmake_minimum_required(VERSION 3.16)
project(autonomous-rc-car LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(steer_test tests/steer_test.cpp)

target_include_directories(steer_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_options(steer_test PRIVATE -Wall -Wextra)

add_executable(throttle_test tests/throttle_test.cpp)
target_include_directories(throttle_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(throttle_test PRIVATE -Wall -Wextra)
target_link_libraries(throttle_test PRIVATE Threads::Threads)
