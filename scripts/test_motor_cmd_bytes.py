#!/usr/bin/env python3
"""Try ALL command bytes for motor control.

Servo uses 0xFF. Motor might use a completely different command.
"""
import smbus
import time

bus = smbus.SMBus(1)
addr = 0x17

def send3(cmd, data):
    for _ in range(3):
        try:
            bus.write_i2c_block_data(addr, cmd, data)
        except:
            pass
        time.sleep(0.02)

print("=== Testing ALL command bytes (0x00-0xFE) for motor ===")
print("This tests each command byte with [0, 150] (channel 0, speed 150)")
print("Watch for wheel movement!\n")

# Skip 0xFF since we know that's servo
for cmd in range(0x00, 0xFF):
    if cmd == 0xFF:
        continue

    print(f"Cmd 0x{cmd:02X}...", end=" ", flush=True)

    # Try with channel 0, value 150
    send3(cmd, [0, 150])
    time.sleep(0.5)

    # Try to stop (just in case it worked)
    send3(cmd, [0, 100])
    send3(0xFF, [0, 100])  # Also try 0xFF stop

    print("done")
    time.sleep(0.2)

print("\n=== Also trying single-byte writes ===")
for val in range(0x00, 0x100):
    if val % 16 == 0:
        print(f"Single byte 0x{val:02X}-0x{min(val+15, 0xFF):02X}...", end=" ", flush=True)
    try:
        bus.write_byte(addr, val)
    except:
        pass
    time.sleep(0.05)
    if val % 16 == 15:
        print("done")

print("\n=== Done ===")
