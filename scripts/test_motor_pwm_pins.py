#!/usr/bin/env python3
"""Test if motor is controlled by PWM pins instead of I2C.

Pi 5 hardware PWM pins: GPIO 12, 13, 18, 19
The motor driver might need a PWM signal directly from the Pi.
"""
import lgpio
import time

h = lgpio.gpiochip_open(0)

# Hardware PWM capable pins on Pi
PWM_PINS = [12, 13, 18, 19]

# Also test some other common pins
OTHER_PINS = [4, 5, 6, 17, 22, 23, 24, 27]

print("=== Testing PWM output for motor control ===\n")

def test_pwm(pin, freq=1000, duty=50):
    """Test PWM on a pin."""
    print(f"GPIO {pin}: PWM {freq}Hz, {duty}% duty...", end=" ", flush=True)
    try:
        lgpio.tx_pwm(h, pin, freq, duty)
        time.sleep(1.5)
        lgpio.tx_pwm(h, pin, freq, 0)  # Stop
        print("done")
    except Exception as e:
        print(f"error: {e}")
    time.sleep(0.3)

print("--- Hardware PWM pins ---")
for pin in PWM_PINS:
    test_pwm(pin)

print("\n--- Software PWM on other pins ---")
for pin in OTHER_PINS:
    test_pwm(pin)

print("\n--- Testing different duty cycles on GPIO 18 (common PWM) ---")
for duty in [25, 50, 75, 100]:
    print(f"GPIO 18: {duty}% duty...", end=" ", flush=True)
    try:
        lgpio.tx_pwm(h, 18, 1000, duty)
        time.sleep(1)
        lgpio.tx_pwm(h, 18, 1000, 0)
        print("done")
    except Exception as e:
        print(f"error: {e}")
    time.sleep(0.3)

lgpio.gpiochip_close(h)
print("\n=== Done ===")
