#!/usr/bin/env python3
"""Test motor control via GPIO pins (H-bridge style).

H-bridges typically use:
- IN1/IN2: Direction control (one HIGH, one LOW = forward/reverse)
- EN/PWM: Speed control (PWM signal)

This script tries common GPIO pins that might control the motor driver.
"""
import lgpio
import time

h = lgpio.gpiochip_open(0)

# Common GPIO pins that might be used for motor control
# Avoiding: 2,3 (I2C), 14,15 (UART), 9,10,25 (LEDs we found)
TEST_PINS = [4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]

print("=== Setting up GPIO pins ===")
for pin in TEST_PINS:
    try:
        lgpio.gpio_claim_output(h, pin, 0)
        print(f"  GPIO {pin}: OK")
    except Exception as e:
        print(f"  GPIO {pin}: {e}")

def set_pin(pin, val):
    try:
        lgpio.gpio_write(h, pin, val)
    except:
        pass

def test_pair(in1, in2, name=""):
    """Test a pair of pins as H-bridge direction control."""
    print(f"\nTesting GPIO {in1} + {in2} as H-bridge {name}")

    # Forward: IN1=HIGH, IN2=LOW
    print(f"  Forward (GPIO{in1}=1, GPIO{in2}=0)...", end=" ", flush=True)
    set_pin(in1, 1)
    set_pin(in2, 0)
    time.sleep(1)
    print("done")

    # Stop
    set_pin(in1, 0)
    set_pin(in2, 0)
    time.sleep(0.3)

    # Reverse: IN1=LOW, IN2=HIGH
    print(f"  Reverse (GPIO{in1}=0, GPIO{in2}=1)...", end=" ", flush=True)
    set_pin(in1, 0)
    set_pin(in2, 1)
    time.sleep(1)
    print("done")

    # Stop
    set_pin(in1, 0)
    set_pin(in2, 0)
    time.sleep(0.3)

print("\n=== Testing GPIO pairs as H-bridge direction pins ===")
print("Watch for motor movement...")

# Test adjacent pin pairs
pairs = [
    (4, 5), (5, 6), (6, 7), (7, 8),
    (11, 12), (12, 13),
    (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24),
    (26, 27),
]

for p1, p2 in pairs:
    test_pair(p1, p2)

# Also try single pins with PWM (in case EN pin just needs to be high)
print("\n=== Testing single pins HIGH (enable pins) ===")
for pin in TEST_PINS:
    print(f"GPIO {pin} HIGH for 1 sec...", end=" ", flush=True)
    set_pin(pin, 1)
    time.sleep(1)
    set_pin(pin, 0)
    print("done")
    time.sleep(0.2)

# Cleanup
for pin in TEST_PINS:
    try:
        lgpio.gpio_free(h, pin)
    except:
        pass
lgpio.gpiochip_close(h)

print("\n=== Done ===")
