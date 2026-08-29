cmake_minimum_required(VERSION 3.16)
project(autonomous-rc-car LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ONNXRUNTIME_ROOT $ENV{HOME}/onnxruntime)

add_library(camera STATIC src/camera.cpp)
target_include_directories(camera PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(camera PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(camera PRIVATE -Wall -Wextra)

add_library(pilotnet_onnx STATIC src/pilotnet_onnx.cpp)
target_include_directories(pilotnet_onnx PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(pilotnet_onnx PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(pilotnet_onnx
                      PUBLIC ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
target_compile_options(pilotnet_onnx PRIVATE -Wall -Wextra)

add_library(stream STATIC src/stream.cpp)
target_include_directories(stream PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(stream PRIVATE -Wall -Wextra)
target_link_libraries(stream PUBLIC Threads::Threads ${OpenCV_LIBS})

add_executable(drive src/main.cpp)
target_link_libraries(drive PRIVATE camera pilotnet_onnx stream Threads::Threads)
target_compile_options(drive PRIVATE -Wall -Wextra)

add_subdirectory(tests)
