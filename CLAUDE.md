# Autonomous RC Car

Self-driving RC car project using a Raspberry Pi 5 and XiaoRGEEK F1 platform.

## Project Overview

This project replaces the proprietary 32-bit XiaoRGEEK Python libraries with pure Python implementations compatible with the 64-bit Raspberry Pi 5.

## Hardware

- **Platform**: XiaoRGEEK F1 RC Car
- **Controller Board**: PWR.A53.A
- **Computer**: Raspberry Pi 5 (64-bit)

## Control Protocols

See `CONTROLS.md` for full details.

### Steering (I2C)

- Address: 0x17
- Command: 0xFF
- Data: [channel=1, value]
- Range: 0 (left) to 170 (right), 85 = center
- Note: Send commands 3x due to buffer quirk

### Motors (GPIO)

- Enable/PWM: GPIO 13
- Direction: GPIO 19 (IN1), GPIO 16 (IN2)
- Forward: IN1=HIGH, IN2=LOW
- Reverse: IN1=LOW, IN2=HIGH

## Key Files

- `main.py` - Entry point
- `src/car/vehicle.py` - Unified vehicle interface (steering, throttle, camera)
- `src/car/actuator.py` - Steering (I2C) and motor (GPIO) control
- `src/car/config.py` - Configuration
- `src/car/datastore.py` - Data collection (recording sessions)
- `src/car/web/` - Tornado web server for manual control
- `src/training/` - Model training code
- `pi/` - Original XiaoRGEEK code backup (reference only)

## Development

### Running on Pi

```bash
cd ~/autonomous-rc-car
uv run python main.py drive
```

### Testing Actuators

```bash
uv run python src/car/actuator.py           # Test steering only
uv run python src/car/actuator.py --throttle  # Test steering and motors
```

### Dependencies

Managed with `uv`. The virtual environment uses `--system-site-packages` to access system-installed packages like `lgpio` and `picamera2`.

## Notes

- The `pi/` directory contains the original 32-bit code for reference - do not modify
- Always test motor changes with the car wheels off the ground
- The I2C controller has a one-command buffer - always send commands 3 times
- The steering servo stays locked after program exit - power cycle the car to release it
