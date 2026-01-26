#!/usr/bin/env python3
"""Test motor with initialization sequences.

The original code uses motorPWM(100) to init at stopped position.
Maybe we need to send an init command first.
"""
import smbus
import time

bus = smbus.SMBus(1)
addr = 0x17

def send3(cmd, data):
    """Send command 3 times (buffer flush trick)."""
    for _ in range(3):
        try:
            bus.write_i2c_block_data(addr, cmd, data)
        except:
            pass
        time.sleep(0.02)

def try_motor(desc, speed=150):
    """Try to run motor at given speed."""
    print(f"  {desc} speed={speed}...", end=" ", flush=True)
    # Channel 0 for motor
    send3(0xFF, [0, speed])
    time.sleep(1)
    # Stop
    send3(0xFF, [0, 100])
    print("done")
    time.sleep(0.3)

print("=== Testing motor with initialization sequences ===\n")

# Test 1: Maybe motor needs to be set to 100 (stopped) first before any movement
print("Test 1: Init to 100 (stopped) then move")
send3(0xFF, [0, 100])
time.sleep(0.5)
try_motor("Forward after init")

# Test 2: Try different command bytes for init
print("\nTest 2: Init with different command bytes then motor")
for init_cmd in [0x00, 0x01, 0x10, 0x80, 0xFE]:
    print(f"Init cmd 0x{init_cmd:02X}...")
    try:
        bus.write_byte(addr, init_cmd)
    except:
        pass
    try:
        bus.write_i2c_block_data(addr, init_cmd, [100])
    except:
        pass
    time.sleep(0.3)
    try_motor("After init")

# Test 3: Maybe motor is channel 2 or 3, not 0
print("\nTest 3: Try channels 0, 2, 3, 4 with init sequence")
for ch in [0, 2, 3, 4]:
    print(f"Channel {ch}:")
    # Init to stopped
    send3(0xFF, [ch, 100])
    time.sleep(0.3)
    # Forward
    print(f"  Forward ch={ch}...", end=" ", flush=True)
    send3(0xFF, [ch, 150])
    time.sleep(1)
    # Stop
    send3(0xFF, [ch, 100])
    print("done")
    time.sleep(0.3)

# Test 4: Values in original config (0, 100, 200)
print("\nTest 4: Original config values (0=rev, 100=stop, 200=fwd)")
for ch in [0, 2, 3]:
    print(f"Channel {ch}:")
    send3(0xFF, [ch, 100])  # init stopped
    time.sleep(0.3)

    print(f"  Reverse (0)...", end=" ", flush=True)
    send3(0xFF, [ch, 0])
    time.sleep(1)
    send3(0xFF, [ch, 100])
    print("done")
    time.sleep(0.3)

    print(f"  Forward (200)...", end=" ", flush=True)
    send3(0xFF, [ch, 200])
    time.sleep(1)
    send3(0xFF, [ch, 100])
    print("done")
    time.sleep(0.3)

# Test 5: Maybe motor uses negative numbers for reverse
print("\nTest 5: Try signed values (-100 to 100 mapped to 0-200)")
send3(0xFF, [0, 100])  # init
time.sleep(0.3)
for val in [50, 150, 0, 200, 255]:
    print(f"  Value {val}...", end=" ", flush=True)
    send3(0xFF, [0, val])
    time.sleep(1)
    send3(0xFF, [0, 100])
    print("done")
    time.sleep(0.3)

print("\n=== Done ===")
