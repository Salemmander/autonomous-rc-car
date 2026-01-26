#!/usr/bin/env python3
"""Try writing to specific registers for motor control."""
import smbus
import time

bus = smbus.SMBus(1)
addr = 0x17

def send3(reg, data):
    for _ in range(3):
        try:
            bus.write_i2c_block_data(addr, reg, data)
        except:
            pass
        time.sleep(0.02)

def write_byte3(reg, val):
    for _ in range(3):
        try:
            bus.write_byte_data(addr, reg, val)
        except:
            pass
        time.sleep(0.02)

print("=== Testing writes to specific registers ===\n")

# Registers 0x12, 0x13, 0x14 had unique values - try writing to them
print("--- Writing to registers 0x12, 0x13, 0x14 ---")
for reg in [0x12, 0x13, 0x14]:
    for val in [0, 50, 100, 150, 200]:
        print(f"Reg 0x{reg:02X} = {val}...", end=" ", flush=True)
        write_byte3(reg, val)
        time.sleep(0.8)
        write_byte3(reg, 0)
        print("done")
        time.sleep(0.2)

# Try register 0x00 (device ID register) - maybe it's also a control register
print("\n--- Writing to register 0x00 ---")
for val in [0, 50, 100, 150, 200]:
    print(f"Reg 0x00 = {val}...", end=" ", flush=True)
    write_byte3(0x00, val)
    time.sleep(0.8)
    print("done")
    time.sleep(0.2)

# Maybe motor uses registers 0x02-0x05 (after channel 1 for servo)
print("\n--- Writing to registers 0x02-0x05 ---")
for reg in [0x02, 0x03, 0x04, 0x05]:
    print(f"Reg 0x{reg:02X} = 150...", end=" ", flush=True)
    write_byte3(reg, 150)
    time.sleep(0.8)
    write_byte3(reg, 100)
    print("done")
    time.sleep(0.2)

# Try writing [direction, speed] to different base registers
print("\n--- Writing [direction, speed] pairs ---")
for reg in [0x00, 0x02, 0x10, 0x20]:
    print(f"Reg 0x{reg:02X} [1, 150]...", end=" ", flush=True)
    send3(reg, [1, 150])
    time.sleep(0.8)
    send3(reg, [0, 0])
    print("done")
    time.sleep(0.2)

print("\n=== Done ===")
