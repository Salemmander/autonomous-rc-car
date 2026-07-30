#include "vehicle.h"

#include <chrono>
#include <iostream>
#include <thread>

int main() {
    // Wheels off the ground before running.
    Vehicle vehicle{1280, 720};

    if (!vehicle.ok()) {
        std::cerr << "Failed to open vehicle (steering and/or throttle)\n";
        return 1;
    }

    std::cout << "stop / center\n";
    vehicle.stop();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "forward + right\n";
    vehicle.drive(0.5f, 0.3f);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "stop / center\n";
    vehicle.stop();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "reverse + left\n";
    vehicle.drive(-0.5f, -0.3f);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "stop / center\n";
    vehicle.stop();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
