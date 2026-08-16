#include "pilotnet_onnx.h"
#include "vehicle.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

namespace {

    constexpr int WIDTH = 1280;
    constexpr int HEIGHT = 720;
    constexpr auto MODEL_PATH = "models/pilotnet.onnx";

    std::atomic<bool> g_running{true};

    void on_sigint(int) {
        g_running = false;
    }

} // namespace

int main() {
    std::signal(SIGINT, on_sigint);

    Vehicle car{WIDTH, HEIGHT};
    if (!car.ok()) {
        std::cerr << "Failed to open vehicle\n";
        return 1;
    }

    PilotNetOnnx net{MODEL_PATH, HEIGHT, WIDTH};
    car.start();

    float steering = 0.0f;
    float throttle = 0.0f;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        const auto frame = car.getFrame();
        if (!net.infer(frame, steering, throttle)) {
            continue;
        }
        car.drive(steering, throttle);
    }

    car.stop();
    std::cout << "Car stopped\n";
    return 0;
}
