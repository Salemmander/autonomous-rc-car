#pragma once

#include "steering.h"
#include "throttle.h"

class Vehicle {
private:
    SteeringController steeringctl;
    ThrottleController throttlectl;

    float current_steering_angle{0.0};
    float current_throttle{0.0};

public:
    ~Vehicle() {
        stop();
    }

    bool ok() const {
        return steeringctl.ok() && throttlectl.ok();
    }
    void drive(float angle, float throttle) {
        if (!ok()) {
            return;
        }
        steeringctl.setSteeringAngle(angle);
        throttlectl.setThrottle(throttle);

        current_steering_angle = angle;
        current_throttle = throttle;
    }

    void stop() {
        drive(0.0, 0.0);
    }
};
