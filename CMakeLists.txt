cmake_minimum_required(VERSION 3.16)
project(autonomous-rc-car LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(camera STATIC src/camera.cpp)
target_include_directories(camera PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(camera PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(camera PRIVATE -Wall -Wextra)

add_executable(drive src/main.cpp)
set(ONNXRUNTIME_ROOT $ENV{HOME}/onnxruntime)
target_link_libraries(drive PRIVATE camera Threads::Threads
                                    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
target_compile_options(drive PRIVATE -Wall -Wextra)
target_include_directories(drive PRIVATE ${ONNXRUNTIME_ROOT}/include)

add_subdirectory(tests)
