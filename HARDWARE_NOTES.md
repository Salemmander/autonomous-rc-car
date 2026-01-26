# XiaoR GEEK F1 Donkey Car - Hardware Notes

## Overview

This is a XiaoR GEEK F1 RC car kit that came pre-configured with Donkey Car software for autonomous driving. The SD card contains Raspbian GNU/Linux 10 (Buster) with pre-installed machine learning libraries and a trained self-driving model.

## Quick Start

1. **Power on the car** and wait 30-60 seconds for it to boot (LEDs will flash when ready)

2. **Connect to the car's WiFi** from your laptop:
   - Look for network `XiaoRGEEK_F1_Pi4_*`
   - No password required (open network)
   - You will lose internet access while connected

3. **Open the web interface** at `https://192.168.1.1:8887`
   - You'll get a certificate warning - click through it (self-signed cert)
   - You should see the camera feed and controls

4. **Switch to manual mode** - the car starts in autonomous mode by default, so switch to "user" mode in the web interface before it drives off

5. **Optional - SSH in to change password:**
   ```bash
   ssh pi@192.168.1.1
   # default password: raspberry
   passwd  # change it
   ```

## Security Considerations

The default configuration has some security weaknesses:

1. **Open WiFi** - The hotspot has no password. Anyone nearby can connect.

2. **Default credentials** - The `pi` user password is likely still `raspberry`

3. **Multiple remote access services enabled:**
   - SSH (port 22)
   - VNC
   - XRDP (remote desktop)

**Recommended fixes:**

1. Change the `pi` user password:
   ```bash
   ssh pi@192.168.1.1
   passwd
   ```

2. Add WPA2 to `/etc/hostapd/hostapd.conf`:
   ```
   wpa=2
   wpa_passphrase=YOUR_PASSWORD
   wpa_key_mgmt=WPA-PSK
   ```

3. Disable services you don't need:
   ```bash
   sudo systemctl disable xrdp
   sudo systemctl disable vncserver-x11-serviced
   ```

## Hardware Components

### Main Board
- **Raspberry Pi 4** (based on kernel images present)
- **XiaoR GEEK PWR.A53 Driver Board** - Custom HAT for motor/servo control

### Motor Control
- **PCA9685** - 16-channel I2C PWM controller at address `0x40`
  - Channel 0: Throttle/ESC control
  - Channel 1: Steering servo
- **L298N-style H-Bridge** - DC motor driver with ENA/ENB enable pins

### GPIO Pin Assignments (BCM numbering)
| Pin | Function |
|-----|----------|
| 10  | LED0     |
| 9   | LED1     |
| 25  | LED2     |

### PWM Calibration Values (from config.py)
```python
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 40
STEERING_RIGHT_PWM = 150

THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 200
THROTTLE_STOPPED_PWM = 100
THROTTLE_REVERSE_PWM = 0
```

## Network Configuration

On boot, the Pi creates its own WiFi access point:
- **SSID**: `XiaoRGEEK_F1_Pi4_XXXXXX` (last 6 chars of MAC address)
- **IP Address**: `192.168.1.1`
- **Web Interface**: `https://192.168.1.1:8887`

This is configured in `/home/pi/work/ap.sh` and runs automatically via `/etc/rc.local`.

## Software Stack

### Pre-installed Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 1.13.1 | Neural network inference |
| Keras | 2.2.4 | High-level ML API |
| OpenCV | 4.4.0 | Computer vision |
| NumPy | 1.19.4 | Numerical computing |
| Pandas | 1.2.0 | Data manipulation |

### Donkey Car Framework
Located at: `pi/home/pi/work/donkeycar-pi-2.5.8/`

Key files:
- `pi/home/pi/mycar/manage.py` - Main control script
- `pi/home/pi/mycar/config.py` - Car configuration
- `pi/home/pi/mycar/models/mypilot` - Pre-trained model

### Proprietary Components
The following are compiled `.so` files from XiaoR GEEK (no source available):
- `actuator.so` - Motor/servo control
- `xrcamera.so` - Camera interface
- `_XiaoRGEEK_SERVO_.so` - Servo control library

## Using the Existing Software

### Manual Driving (Web Interface)
```bash
cd ~/mycar
python3 manage.py drive
```
Then open `https://192.168.1.1:8887` in a browser connected to the Pi's WiFi.

### Manual Driving (Joystick)
```bash
python3 ~/mycar/manage.py drive --js
```

### Autonomous Driving
```bash
python3 ~/mycar/manage.py drive --model ~/mycar/models/mypilot
```

### Training a New Model
1. Drive manually with recording enabled (data saves to `~/mycar/tub/`)
2. Train: `python3 ~/mycar/manage.py train --model ~/mycar/models/my_new_model`
3. Run: `python3 ~/mycar/manage.py drive --model ~/mycar/models/my_new_model`

## Direct Hardware Control

### Using Standard Python Libraries

You can bypass the proprietary `.so` files using standard I2C/GPIO libraries:

```python
import smbus
import RPi.GPIO as GPIO
import time

# PCA9685 I2C address and registers
PCA9685_ADDR = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

bus = smbus.SMBus(1)

def init_pca9685(freq=50):
    """Initialize PCA9685 for 50Hz (standard servo frequency)"""
    prescale = int(25000000.0 / (4096 * freq) - 1)
    bus.write_byte_data(PCA9685_ADDR, MODE1, 0x10)  # sleep
    bus.write_byte_data(PCA9685_ADDR, PRESCALE, prescale)
    bus.write_byte_data(PCA9685_ADDR, MODE1, 0x00)  # wake
    time.sleep(0.005)
    bus.write_byte_data(PCA9685_ADDR, MODE1, 0xA0)  # auto-increment

def set_pwm(channel, pulse):
    """Set PWM pulse (0-4095) on channel"""
    reg = LED0_ON_L + 4 * channel
    bus.write_byte_data(PCA9685_ADDR, reg, 0)
    bus.write_byte_data(PCA9685_ADDR, reg + 1, 0)
    bus.write_byte_data(PCA9685_ADDR, reg + 2, pulse & 0xFF)
    bus.write_byte_data(PCA9685_ADDR, reg + 3, pulse >> 8)

def steering(value):
    """Set steering: -1.0 (left) to 1.0 (right)"""
    # Map -1..1 to LEFT_PWM..RIGHT_PWM
    left_pwm, right_pwm = 40, 150
    pulse = int(((value + 1) / 2) * (right_pwm - left_pwm) + left_pwm)
    set_pwm(1, pulse)

def throttle(value):
    """Set throttle: -1.0 (reverse) to 1.0 (forward)"""
    # Map -1..1 to REVERSE_PWM..FORWARD_PWM
    min_pwm, zero_pwm, max_pwm = 0, 100, 200
    if value >= 0:
        pulse = int(zero_pwm + value * (max_pwm - zero_pwm))
    else:
        pulse = int(zero_pwm + value * (zero_pwm - min_pwm))
    set_pwm(0, pulse)

# Example usage
init_pca9685(50)
steering(0)      # Center
throttle(0.25)   # 25% forward
```

### Alternative: Adafruit PCA9685 Library
```bash
pip install adafruit-circuitpython-pca9685 adafruit-circuitpython-servokit
```

```python
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)
kit.servo[1].angle = 90       # Steering center (0-180)
kit.continuous_servo[0].throttle = 0.25  # Throttle
```

### LED Control
```python
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(10, GPIO.OUT)  # LED0
GPIO.setup(9, GPIO.OUT)   # LED1
GPIO.setup(25, GPIO.OUT)  # LED2

GPIO.output(10, GPIO.LOW)   # LED on
GPIO.output(10, GPIO.HIGH)  # LED off
```

## File Structure

```
pi/
├── boot/                    # Boot partition (kernel, device trees, config)
│   ├── config.txt          # Pi boot configuration
│   ├── cmdline.txt         # Kernel command line
│   └── kernel*.img         # Linux kernels
├── home/pi/
│   ├── mycar/              # Donkey Car project
│   │   ├── manage.py       # Main control script
│   │   ├── config.py       # Car configuration
│   │   ├── models/         # Trained models
│   │   │   └── mypilot     # Pre-trained model (3.1MB)
│   │   └── tub/            # Training data
│   └── work/
│       ├── donkeycar-pi-2.5.8/  # Donkey Car framework
│       ├── ap.sh           # WiFi AP setup script
│       ├── start.sh        # Auto-start script
│       ├── init_led.py     # LED initialization
│       └── *.whl           # Pre-compiled Python wheels
└── etc/
    └── rc.local            # Startup script (runs ap.sh and start.sh)
```

## Notes

- The OS is dated (Raspbian Buster from Aug 2022). Consider upgrading to Raspberry Pi OS Bookworm for newer features.
- The pre-trained model was trained on a specific track. You'll need to retrain for your environment.
- PWM values may need calibration for your specific servos/ESC.
- The camera uses a custom `XRCamera` class. Standard PiCamera should work as a replacement.

## Resources

- [Donkey Car Documentation](https://docs.donkeycar.com/)
- [PCA9685 Datasheet](https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf)
- [XiaoR GEEK Website](http://www.xiao-r.com/) (Chinese)
