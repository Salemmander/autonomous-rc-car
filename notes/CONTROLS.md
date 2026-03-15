# XiaoRGEEK PWR.A53.A Control Protocol

How to control the steering servo and DC motors on the XiaoRGEEK F1 RC car with a Raspberry Pi 5.

## Steering (I2C Servo)

| Parameter   | Value                                        |
| ----------- | -------------------------------------------- |
| I2C Address | 0x17                                         |
| Command     | 0xFF                                         |
| Data        | [channel=1, value]                           |
| Range       | 0 (full left), 85 (center), 170 (full right) |

Send commands 3x — the controller has a one-command buffer that delays execution otherwise.

The servo stays locked after program exit. Power cycle the car to release it.

## Motors (GPIO H-Bridge)

### Motor 1 (M1)

| Pin          | GPIO |
| ------------ | ---- |
| Enable (PWM) | 13   |
| IN1          | 19   |
| IN2          | 16   |

| Direction | IN1  | IN2  |
| --------- | ---- | ---- |
| Forward   | HIGH | LOW  |
| Reverse   | LOW  | HIGH |
| Stop      | LOW  | LOW  |

## Code

See `src/car/actuator.py` for `PWMSteering` and `PWMThrottle` classes that map -1.0..1.0 to hardware values.
