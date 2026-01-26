#!/usr/bin/env python3
"""Read all I2C registers from 0x17 to understand the device."""
import smbus
import time

bus = smbus.SMBus(1)
addr = 0x17

print("=== Reading I2C registers from 0x17 ===\n")

# Try reading single bytes from different register addresses
print("Single byte reads (register 0x00-0xFF):")
for reg in range(0x00, 0x100):
    try:
        val = bus.read_byte_data(addr, reg)
        if val != 0 and val != 0xFF:  # Only show non-zero, non-0xFF
            print(f"  Reg 0x{reg:02X} = 0x{val:02X} ({val})")
    except:
        pass
    time.sleep(0.01)

print("\n--- Block reads ---")
# Try reading blocks of data
for reg in [0x00, 0x01, 0x10, 0x20, 0x40, 0x80, 0xFF]:
    try:
        data = bus.read_i2c_block_data(addr, reg, 8)
        if any(d != 0 and d != 0xFF for d in data):
            print(f"  Reg 0x{reg:02X}: {[hex(d) for d in data]}")
    except Exception as e:
        print(f"  Reg 0x{reg:02X}: read error - {e}")
    time.sleep(0.05)

print("\n--- Try reading without register (raw read) ---")
try:
    # Read multiple bytes directly
    data = bus.read_i2c_block_data(addr, 0, 16)
    print(f"  Raw 16 bytes: {[hex(d) for d in data]}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Done ===")
