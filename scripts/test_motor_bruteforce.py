#!/usr/bin/env python3
"""Brute force motor commands with direction+speed patterns."""
import smbus
import time

bus = smbus.SMBus(1)
addr = 0x17

def send(cmd, data):
    try:
        bus.write_i2c_block_data(addr, cmd, data)
        bus.write_i2c_block_data(addr, cmd, data)
        bus.write_i2c_block_data(addr, cmd, data)
        return True
    except:
        return False

def test(desc, cmd, data, duration=0.8):
    print(f"{desc}: 0x{cmd:02X} {data}...", end=" ", flush=True)
    if send(cmd, data):
        print("sent")
        time.sleep(duration)
        # Try to stop
        stop_data = [data[0], 0] if len(data) >= 2 else [0]
        send(cmd, stop_data)
    else:
        print("error")
    time.sleep(0.2)

print("=== Motor brute force test ===")
print("Watch wheels carefully!\n")

# Pattern 1: 0xFF with [motor, direction, speed]
print("--- 0xFF with 3-byte [id, direction, speed] ---")
for motor_id in [0, 2, 3, 4]:
    for direction in [0, 1, 2]:
        test(f"Motor {motor_id} dir={direction}", 0xFF, [motor_id, direction, 100])

# Pattern 2: 0xFF with [direction_flag + motor, speed]
print("\n--- 0xFF with direction encoded in first byte ---")
for first_byte in [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x20, 0x21]:
    test(f"First=0x{first_byte:02X}", 0xFF, [first_byte, 100])

# Pattern 3: Motor-specific command bytes (0x01-0x0F)
print("\n--- Command bytes 0x01-0x0F ---")
for cmd in range(0x01, 0x10):
    test(f"Cmd 0x{cmd:02X}", cmd, [100])
    test(f"Cmd 0x{cmd:02X} [0,100]", cmd, [0, 100])

# Pattern 4: High command bytes
print("\n--- High command bytes ---")
for cmd in [0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0xFE]:
    test(f"Cmd 0x{cmd:02X}", cmd, [100])
    test(f"Cmd 0x{cmd:02X} [0,100]", cmd, [0, 100])

# Pattern 5: XiaoRGEEK might use M1=channel 3 or 4
print("\n--- Higher channels 3-10 ---")
for ch in range(3, 11):
    test(f"Ch {ch}", 0xFF, [ch, 100])

# Pattern 6: Write raw byte sequences
print("\n--- Raw byte sequences ---")
for seq in [[0x01, 0x01, 100], [0x02, 0x01, 100], [0x03, 0x01, 100],
            [0x01, 100, 0x01], [0x02, 100, 0x01],
            [100, 0, 0], [0, 100, 0], [0, 0, 100],
            [1, 1, 1, 100], [0, 0, 0, 100]]:
    print(f"Seq {seq}...", end=" ", flush=True)
    try:
        bus.write_i2c_block_data(addr, seq[0], seq[1:])
        bus.write_i2c_block_data(addr, seq[0], seq[1:])
        bus.write_i2c_block_data(addr, seq[0], seq[1:])
        print("sent")
        time.sleep(0.8)
    except Exception as e:
        print(f"err: {e}")
    time.sleep(0.2)

print("\n=== Done ===")
