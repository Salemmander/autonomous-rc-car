#include "throttle.h"

#include <chrono>
#include <iostream>
#include <thread>

int main() {
    // Wheels off the ground before running.
    ThrottleController throttle;

    if (!throttle.ok()) {
        std::cerr << "Failed to open GPIO throttle\n";
        return 1;
    }

    std::cout << "stop\n";
    throttle.setThrottle(0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "forward 0.3\n";
    throttle.setThrottle(0.3f);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "stop\n";
    throttle.setThrottle(0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "reverse 0.3\n";
    throttle.setThrottle(-0.3f);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "stop\n";
    throttle.setThrottle(0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
