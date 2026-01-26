# XiaoRGEEK PWR.A53.A Control Protocol

This document describes how to control the steering servo and DC motors on the XiaoRGEEK F1 RC car platform with a Raspberry Pi 5.

## Hardware Overview

- **Board**: XiaoRGEEK PWR.A53.A driver board
- **I2C Address**: 0x17
- **Steering**: Servo connected to channel 1 (3-pin header: red/yellow/brown)
- **Motors**: DC motors connected to M1 screw terminals (green wires)

## Steering Control (I2C)

The steering servo is controlled via I2C.

| Parameter | Value |
|-----------|-------|
| I2C Address | 0x17 |
| Command Byte | 0xFF |
| Channel | 1 |
| Value Range | 0-170 |

### Protocol

Send I2C block data: `[command_byte, channel, value]`

```python
import smbus

bus = smbus.SMBus(1)
address = 0x17

# Set steering position (0=left, 85=center, 170=right)
bus.write_i2c_block_data(address, 0xFF, [1, 85])
```

### Buffer Quirk

The controller has a one-command buffer. Commands are delayed by one unless sent multiple times. Always send commands 3x for immediate execution:

```python
def set_steering(value):
    cmd = [1, value]
    bus.write_i2c_block_data(address, 0xFF, cmd)
    bus.write_i2c_block_data(address, 0xFF, cmd)
    bus.write_i2c_block_data(address, 0xFF, cmd)
```

### Steering Values

| Position | Value |
|----------|-------|
| Full Left | 0 |
| Center | 85 |
| Full Right | 170 |

## Motor Control (GPIO)

The DC motors are controlled via GPIO pins using an H-bridge driver on the board.

### GPIO Pins (Motor 1 / M1)

| Function | GPIO Pin |
|----------|----------|
| Enable (PWM) | 13 |
| IN1 (Direction) | 19 |
| IN2 (Direction) | 16 |

### Direction Control

| Direction | IN1 | IN2 |
|-----------|-----|-----|
| Forward | HIGH | LOW |
| Reverse | LOW | HIGH |
| Stop | LOW | LOW |

### Protocol

```python
import lgpio

h = lgpio.gpiochip_open(0)

# Setup pins
lgpio.gpio_claim_output(h, 13, 0)  # ENA
lgpio.gpio_claim_output(h, 19, 0)  # IN1
lgpio.gpio_claim_output(h, 16, 0)  # IN2

# Forward at 75% speed
lgpio.gpio_write(h, 19, 1)  # IN1 = HIGH
lgpio.gpio_write(h, 16, 0)  # IN2 = LOW
lgpio.tx_pwm(h, 13, 100, 75)  # 100Hz, 75% duty cycle

# Stop
lgpio.gpio_write(h, 19, 0)
lgpio.gpio_write(h, 16, 0)
lgpio.tx_pwm(h, 13, 100, 0)
```

## Motor 2 Pins (Optional)

If using M2 terminals for a second motor:

| Function | GPIO Pin |
|----------|----------|
| Enable (PWM) | 20 |
| IN3 (Direction) | 21 |
| IN4 (Direction) | 26 |

## Python Classes

See `actuator.py` for ready-to-use classes:

- `PWMSteering`: Steering control, maps -1.0 to 1.0 to servo position
- `PWMThrottle`: Motor control, maps -1.0 to 1.0 to motor speed/direction

### Usage

```python
from actuator import PWMSteering, PWMThrottle

steering = PWMSteering()
throttle = PWMThrottle()

# Steer right
steering.run(0.5)

# Drive forward at 50%
throttle.run(0.5)

# Stop
throttle.run(0)
steering.run(0)
```

## References

- [Kuman SM9 / PWR.A53.A Android Things Driver](https://github.com/leinardi/androidthings-kuman-sm9)
