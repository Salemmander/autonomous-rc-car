"""
Configuration for Raspberry Pi 5

PWM values calibrated for XiaoR GEEK F1 hardware.
"""

import os

# Paths
CAR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Vehicle loop
DRIVE_LOOP_HZ = 20

# Camera (Pi 5 with picamera2)
CAMERA_RESOLUTION = (720, 1280)  # (height, width)

# Data collection
DATA_PATH = os.path.join(CAR_PATH, "data")
RECORD_FPS = 10
JPEG_QUALITY = 70

# Steering - I2C servo on channel 1
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 0
STEERING_RIGHT_PWM = 170
