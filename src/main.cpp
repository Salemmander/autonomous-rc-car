#include "vehicle.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

#define WIDTH 1280
#define HEIGHT 720

std::atomic<bool> running{true};

void signal_handler(int) {
    running = false;
}

int main() {
    // Wheels off the ground when testing throttle. Ctrl+C to stop.
    std::signal(SIGINT, signal_handler);

    Vehicle car{WIDTH, HEIGHT};
    if (!car.ok()) {
        std::cerr << "Failed to open vehicle\n";
        return 1;
    }

    car.start();

    while (running) {
        car.drive(0.0, 0.5);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    car.stop();
    std::cout << "Car stopped\n";
    return 0;
}
