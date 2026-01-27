#!/usr/bin/env python3
"""Test different ways to release the servo."""
import smbus
import time

bus = smbus.SMBus(1)
addr = 0x17

def send(cmd, data):
    for _ in range(3):
        try:
            bus.write_i2c_block_data(addr, cmd, data)
        except:
            pass
        time.sleep(0.02)

print("First, move servo to center...")
send(0xFF, [1, 85])
time.sleep(1)

print("\nTrying different release methods:")
print("After each test, try to move the servo by hand.\n")

input("Press Enter to try value 255...")
send(0xFF, [1, 255])
print("Sent [1, 255]")

input("Press Enter to try command 0x00...")
send(0x00, [1, 0])
print("Sent cmd=0x00 [1, 0]")

input("Press Enter to try command 0xFE...")
send(0xFE, [1, 0])
print("Sent cmd=0xFE [1, 0]")

input("Press Enter to try single byte 0x00...")
try:
    bus.write_byte(addr, 0x00)
except:
    pass
print("Sent single byte 0x00")

input("Press Enter to close I2C bus...")
bus.close()
print("Bus closed")

print("\nDone. If servo is still stuck, it needs a power cycle to release.")
