#!/usr/bin/env python3
"""Scan all I2C addresses and try motor commands on each."""
import smbus
import time

bus = smbus.SMBus(1)

print("=== Scanning I2C bus for all devices ===")
found = []
for addr in range(0x03, 0x78):
    try:
        bus.read_byte(addr)
        found.append(addr)
        print(f"  Found device at 0x{addr:02X}")
    except:
        pass

if not found:
    print("No I2C devices found!")
    exit(1)

print(f"\nFound {len(found)} device(s): {[hex(a) for a in found]}")
print()

# Try motor commands on each address (except 0x17 which we know is servo)
for addr in found:
    if addr == 0x17:
        print(f"Skipping 0x{addr:02X} (known servo controller)")
        continue

    print(f"\n=== Testing address 0x{addr:02X} ===")

    # Try various command patterns
    patterns = [
        ("0xFF [0, 100]", 0xFF, [0, 100]),
        ("0xFF [1, 100]", 0xFF, [1, 100]),
        ("0xFF [2, 100]", 0xFF, [2, 100]),
        ("0x00 [100]", 0x00, [100]),
        ("0x01 [100]", 0x01, [100]),
        ("0x02 [100]", 0x02, [100]),
        ("0x10 [100]", 0x10, [100]),
        ("0x82 [100]", 0x82, [100]),  # TB6612 style
        ("0x83 [100]", 0x83, [100]),
    ]

    for name, cmd, data in patterns:
        print(f"  Trying {name}...", end=" ", flush=True)
        try:
            bus.write_i2c_block_data(addr, cmd, data)
            bus.write_i2c_block_data(addr, cmd, data)
            bus.write_i2c_block_data(addr, cmd, data)
            print("sent")
            time.sleep(0.8)
            # Stop
            bus.write_i2c_block_data(addr, cmd, [0])
        except Exception as e:
            print(f"error: {e}")
        time.sleep(0.2)

print("\n=== Done ===")
