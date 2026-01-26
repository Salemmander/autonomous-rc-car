#!/usr/bin/env python3
"""Test motor control using correct GPIO pins from PWR.A53.A documentation.

Motor 1 (M1):
  - Enable/PWM: GPIO 13
  - Direction: GPIO 19 (IN1), GPIO 16 (IN2)

Motor 2 (M2):
  - Enable/PWM: GPIO 20
  - Direction: GPIO 21 (IN3), GPIO 26 (IN4)

Direction control:
  - Forward:  IN1=HIGH, IN2=LOW
  - Backward: IN1=LOW, IN2=HIGH
  - Stop:     IN1=LOW, IN2=LOW (or IN1=HIGH, IN2=HIGH)
"""
import lgpio
import time

h = lgpio.gpiochip_open(0)

# Motor 1 pins
M1_ENA = 13  # PWM/Enable
M1_IN1 = 19  # Direction
M1_IN2 = 16  # Direction

# Motor 2 pins
M2_ENB = 20  # PWM/Enable
M2_IN3 = 21  # Direction
M2_IN4 = 26  # Direction

ALL_PINS = [M1_ENA, M1_IN1, M1_IN2, M2_ENB, M2_IN3, M2_IN4]

print("=== Motor GPIO Test (PWR.A53.A pins) ===\n")

# Setup all pins as outputs
print("Setting up GPIO pins...")
for pin in ALL_PINS:
    try:
        lgpio.gpio_claim_output(h, pin, 0)
        print(f"  GPIO {pin}: OK")
    except Exception as e:
        print(f"  GPIO {pin}: {e}")

def motor1_forward(speed=100):
    """Motor 1 forward."""
    lgpio.gpio_write(h, M1_IN1, 1)
    lgpio.gpio_write(h, M1_IN2, 0)
    lgpio.tx_pwm(h, M1_ENA, 100, speed)

def motor1_backward(speed=100):
    """Motor 1 backward."""
    lgpio.gpio_write(h, M1_IN1, 0)
    lgpio.gpio_write(h, M1_IN2, 1)
    lgpio.tx_pwm(h, M1_ENA, 100, speed)

def motor1_stop():
    """Motor 1 stop."""
    lgpio.gpio_write(h, M1_IN1, 0)
    lgpio.gpio_write(h, M1_IN2, 0)
    lgpio.tx_pwm(h, M1_ENA, 100, 0)

def motor2_forward(speed=100):
    """Motor 2 forward."""
    lgpio.gpio_write(h, M2_IN3, 1)
    lgpio.gpio_write(h, M2_IN4, 0)
    lgpio.tx_pwm(h, M2_ENB, 100, speed)

def motor2_backward(speed=100):
    """Motor 2 backward."""
    lgpio.gpio_write(h, M2_IN3, 0)
    lgpio.gpio_write(h, M2_IN4, 1)
    lgpio.tx_pwm(h, M2_ENB, 100, speed)

def motor2_stop():
    """Motor 2 stop."""
    lgpio.gpio_write(h, M2_IN3, 0)
    lgpio.gpio_write(h, M2_IN4, 0)
    lgpio.tx_pwm(h, M2_ENB, 100, 0)

def all_stop():
    motor1_stop()
    motor2_stop()

print("\n--- Testing Motor 1 ---")
print("Motor 1 Forward...", end=" ", flush=True)
motor1_forward(75)
time.sleep(1.5)
motor1_stop()
print("done")
time.sleep(0.5)

print("Motor 1 Backward...", end=" ", flush=True)
motor1_backward(75)
time.sleep(1.5)
motor1_stop()
print("done")
time.sleep(0.5)

print("\n--- Testing Motor 2 ---")
print("Motor 2 Forward...", end=" ", flush=True)
motor2_forward(75)
time.sleep(1.5)
motor2_stop()
print("done")
time.sleep(0.5)

print("Motor 2 Backward...", end=" ", flush=True)
motor2_backward(75)
time.sleep(1.5)
motor2_stop()
print("done")
time.sleep(0.5)

print("\n--- Testing Both Motors ---")
print("Both Forward...", end=" ", flush=True)
motor1_forward(75)
motor2_forward(75)
time.sleep(1.5)
all_stop()
print("done")
time.sleep(0.5)

print("Both Backward...", end=" ", flush=True)
motor1_backward(75)
motor2_backward(75)
time.sleep(1.5)
all_stop()
print("done")

# Cleanup
print("\nCleaning up...")
for pin in ALL_PINS:
    try:
        lgpio.gpio_free(h, pin)
    except:
        pass
lgpio.gpiochip_close(h)

print("\n=== Done ===")
