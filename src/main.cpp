#include "controller.h"
#include "pilotnet_onnx.h"
#include "stream.h"
#include "vehicle.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
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

enum class Mode {
    Manual,
    Auto
};

int main(int argc, char** argv) {
    Mode mode;
    if (argc >= 2) {
        if (strcmp(argv[1], "--manual") == 0) {
            mode = Mode::Manual;
        } else if (strcmp(argv[1], "--auto") == 0) {
            mode = Mode::Auto;
        } else {
            std::cerr << "Only valid args are --manual and --auto\n";
            return 1;
        }
    } else {
        std::cerr << "Must use either --manual or --auto\n";
        return 1;
    }

    std::signal(SIGINT, on_sigint);

    MjpegStream stream{};

    Vehicle car{WIDTH, HEIGHT};
    if (!car.ok()) {
        std::cerr << "Failed to open vehicle\n";
        return 1;
    }
    std::unique_ptr<XboxController> pad;
    std::unique_ptr<PilotNetOnnx> net;

    if (mode == Mode::Manual) {
        pad = std::make_unique<XboxController>();
        pad->start();
    } else if (mode == Mode::Auto) {
        net = std::make_unique<PilotNetOnnx>(MODEL_PATH, HEIGHT, WIDTH);
    }
    car.start();
    stream.start();

    float steering = 0.0f;
    float throttle = 0.0f;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        const auto frame = car.getFrame();
        stream.send(frame, WIDTH, HEIGHT);

        if (mode == Mode::Manual) {
            pad->get_input(steering, throttle);
        } else if (mode == Mode::Auto) {
            if (!net->infer(frame, steering, throttle)) {
                continue;
            }
        }
        car.drive(steering, throttle);
    }

    car.stop();
    std::cout << "Car stopped\n";
    return 0;
}
