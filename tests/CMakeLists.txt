# Hardware smoke tests (run on the Pi with the right devices available).

# Add a smoke-test executable from tests/<name>.cpp. Configures include path,
# -Wall -Wextra, and links Threads.
function(add_car_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  # Camera/throttle/vehicle use std::thread or mutex; harmless for steer_test.
  target_link_libraries(${name} PRIVATE Threads::Threads)
endfunction()

add_car_test(steer_test)
add_car_test(throttle_test)
add_car_test(vehicle_test)
add_car_test(camera_test)
