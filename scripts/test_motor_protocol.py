#!/usr/bin/env python3
"""Try various I2C protocols for motor control."""
import smbus
import time

bus = smbus.SMBus(1)
addr = 0x17

def send(cmd, data):
    """Send command and print what we're trying."""
    try:
        bus.write_i2c_block_data(addr, cmd, data)
        bus.write_i2c_block_data(addr, cmd, data)
        bus.write_i2c_block_data(addr, cmd, data)
    except Exception as e:
        print(f"  Error: {e}")

def test_pattern(name, cmd, data):
    print(f"{name}: cmd=0x{cmd:02X} data={data}")
    send(cmd, data)
    time.sleep(1)
    # Stop
    send(cmd, [data[0], 0] if len(data) >= 2 else [0])
    time.sleep(0.3)

print("=== Testing motor protocols ===")
print("Watch for wheel movement after each test")
print()

# Try channel 0 (we know channel 1 is steering)
print("--- Channel 0 with 0xFF command ---")
for val in [50, 100, 150, 200]:
    test_pattern(f"Ch0 val={val}", 0xFF, [0, val])

# Try channel 2 and 3
print("\n--- Channels 2 and 3 with 0xFF ---")
for ch in [2, 3]:
    test_pattern(f"Ch{ch} val=100", 0xFF, [ch, 100])

# Try common motor command bytes
print("\n--- Different command bytes, channel 0 ---")
for cmd in [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x20, 0x30, 0x40, 0x50, 0x82, 0x83, 0x84, 0x85]:
    test_pattern(f"Cmd 0x{cmd:02X}", cmd, [0, 100])

# Try motor-specific commands (some controllers use these)
print("\n--- Motor-specific patterns ---")
test_pattern("M1 forward", 0xFF, [0x10, 100])  # M1 might be 0x10
test_pattern("M1 forward", 0xFF, [0x11, 100])
test_pattern("M2 forward", 0xFF, [0x12, 100])
test_pattern("M2 forward", 0xFF, [0x13, 100])

# Try enable + motor
print("\n--- Enable patterns ---")
for enable_cmd in [0x01, 0x10, 0x20, 0xFE]:
    print(f"Enable with 0x{enable_cmd:02X}, then motor")
    try:
        bus.write_byte(addr, enable_cmd)
        time.sleep(0.1)
    except:
        pass
    send(0xFF, [0, 100])
    time.sleep(1)
    send(0xFF, [0, 0])
    time.sleep(0.3)

# Try single byte commands
print("\n--- Single byte motor commands ---")
for val in [0x01, 0x02, 0x10, 0x20, 0x50, 0x80, 0xA0, 0xF0]:
    print(f"Single byte: 0x{val:02X}")
    try:
        bus.write_byte(addr, val)
        time.sleep(0.5)
        bus.write_byte(addr, 0x00)
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.3)

# Try 3-byte data patterns
print("\n--- 3-byte data patterns ---")
test_pattern("3-byte [0,1,100]", 0xFF, [0, 1, 100])
test_pattern("3-byte [1,0,100]", 0xFF, [1, 0, 100])
test_pattern("3-byte [0,100,0]", 0xFF, [0, 100, 0])
test_pattern("3-byte [1,100,0]", 0xFF, [1, 100, 0])

print("\n=== Done ===")
print("If nothing moved, the motor driver might use a completely different protocol")
print("or might not be connected to I2C at all.")
