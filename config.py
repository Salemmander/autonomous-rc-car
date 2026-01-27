"""
Donkey Car Configuration for Raspberry Pi 5

PWM values calibrated for XiaoR GEEK F1 hardware.
"""

import os

# Paths
CAR_PATH = os.path.dirname(os.path.realpath(__file__))

# Vehicle loop
DRIVE_LOOP_HZ = 20
MAX_LOOPS = None  # Run indefinitely

# Camera (Pi 5 with picamera2)
CAMERA_RESOLUTION = (120, 160)  # (height, width)
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

# Steering - I2C servo on channel 1
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 0
STEERING_RIGHT_PWM = 170

# Throttle - GPIO motor control (not used, kept for reference)
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 200
THROTTLE_STOPPED_PWM = 100
THROTTLE_REVERSE_PWM = 0

# Web server
WEB_PORT = 8887

# I2C - XiaoRGEEK controller
PCA9685_I2C_ADDR = 0x17
PCA9685_I2C_BUS = 1
