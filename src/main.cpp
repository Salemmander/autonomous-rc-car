#include "onnxruntime_c_api.h"
#include "vehicle.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <vector>

#define WIDTH 1280
#define HEIGHT 720

std::atomic<bool> running{true};

void signal_handler(int) {
    running = false;
}

int main() {
    std::signal(SIGINT, signal_handler);

    Vehicle car{WIDTH, HEIGHT};
    if (!car.ok()) {
        std::cerr << "Failed to open vehicle\n";
        return 1;
    }

    Ort::Env env{ORT_LOGGING_LEVEL_INFO, "Car"};
    Ort::SessionOptions opts{};

    Ort::Session session{env, "models/pilotnet.onnx", opts};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    car.start();

    int64_t shape[] = {1, 3, HEIGHT, WIDTH};
    const size_t plane = static_cast<size_t>(HEIGHT) * WIDTH;
    std::vector<float> input(3 * plane);
    const char* in_names[] = {"rgb_nchw_0_255"};
    const char* out_names[] = {"steer_throttle"};

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        std::vector<uint8_t> frame = car.getFrame();
        if (frame.size() != 3 * plane)
            continue;

        //
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                auto r = frame[(y * WIDTH + x) * 3 + 0];
                auto g = frame[(y * WIDTH + x) * 3 + 1];
                auto b = frame[(y * WIDTH + x) * 3 + 2];

                input[0 * plane + y * WIDTH + x] = r;
                input[1 * plane + y * WIDTH + x] = g;
                input[2 * plane + y * WIDTH + x] = b;
            }
        }
        auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape, 4);

        auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

        float* y = outputs[0].GetTensorMutableData<float>();

        float steering = y[0];
        float throttle = y[1];

        car.drive(steering, throttle);
    }

    car.stop();
    std::cout << "Car stopped\n";
    return 0;
}
