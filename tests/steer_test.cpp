#include "steering.h"

#include <chrono>
#include <iostream>
#include <thread>

int main() {
    SteeringController steering;

    if (!steering.ok()) {
        std::cerr << "Failed to open I2C steering\n";
        return 1;
    }

    std::cout << "center\n";
    steering.setSteeringAngle(0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "right\n";
    steering.setSteeringAngle(1.0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "left\n";
    steering.setSteeringAngle(-1.0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "center\n";
    steering.setSteeringAngle(0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
