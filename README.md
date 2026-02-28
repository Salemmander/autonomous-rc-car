# Autonomous RC Car

End-to-end self-driving RC car using behavioral cloning on a Raspberry Pi 5.

<!-- TODO: Add demo video/photos -->

## Overview

This project builds a self-driving RC car from scratch. It replaces the proprietary 32-bit XiaoRGEEK Python libraries with pure Python implementations compatible with the 64-bit Raspberry Pi 5, and adds a complete behavioral cloning pipeline:

1. **Drive manually** via a web-based controller while recording camera frames and steering data
2. **Train a neural network** (NVIDIA PilotNet) to predict steering from images
3. **Deploy on the car** for autonomous driving

The model successfully drives the car around a track in both directions after training on ~10k frames.

## How It Works

### Data Collection

The car is driven manually through a web UI served from the Pi. During recording, the front-facing camera captures 120x160 RGB frames at 10 FPS. Each frame is saved as a JPEG alongside a CSV log of timestamped steering and throttle values.

### Training

Images are preprocessed before being fed to the model:

- Crop the top 30% of each frame (removes ceiling/sky)
- Convert from RGB to YCbCr color space (separates luminance from color, improving robustness to lighting changes)
- Normalize pixel values to [0, 1]

The model is trained with MSE loss and the Adam optimizer. Early stopping with a patience of 10 epochs prevents overfitting -- training halts when validation loss stops improving and saves the best checkpoint.

### Inference

On the Pi, the trained model runs in a loop: grab a camera frame, apply the same preprocessing transform, run it through the model, and send the predicted steering value to the servo. Throttle is fixed at the same speed used during data collection.

## Training Results

Trained on ~10,000 frames across 5 driving sessions. Best model selected by validation loss.

| Metric              | Value                              |
| ------------------- | ---------------------------------- |
| Best Val Loss (MSE) | 0.0509                             |
| Best Val MAE        | 0.1723                             |
| Epochs              | 28 (early stopped)                 |
| Dataset             | ~10k frames, 80/20 train/val split |

Training convergence:

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE |
| ----- | ---------- | -------- | --------- | ------- |
| 1     | 0.1834     | 0.0775   | 0.3241    | 0.2118  |
| 5     | 0.0551     | 0.0537   | 0.1818    | 0.1797  |
| 10    | 0.0528     | 0.0529   | 0.1781    | 0.1764  |
| 14    | 0.0498     | 0.0517   | 0.1722    | 0.1723  |
| 18    | 0.0468     | 0.0509   | 0.1665    | 0.1738  |
| 28    | 0.0376     | 0.0591   | 0.1478    | 0.1846  |

Val MAE of 0.17 means the model's average steering error is 0.17 on a [-1, 1] scale.

## Hardware

- **Platform**: XiaoRGEEK F1 RC Car
- **Controller Board**: PWR.A53.A
- **Computer**: Raspberry Pi 5 (64-bit)
- **Camera**: 720p USB webcam (captures at 120x160)
- **Steering**: I2C servo via PWR.A53.A controller
- **Motors**: GPIO-controlled DC motors

## Project Structure

```
autonomous-rc-car/
  main.py                     # Entry point (drive, pilotnet)
  src/
    car/
      vehicle.py              # Unified vehicle interface (steering, throttle, camera)
      actuator.py             # Steering (I2C) and motor (GPIO) control
      config.py               # Hardware configuration
      datastore.py            # Data collection (recording sessions)
      web/                    # Tornado web server for manual control
    training/
      pilotnet.py             # PilotNet model + training entry point
      train.py                # Reusable Trainer class
      dataset.py              # DrivingDataset (PyTorch Dataset)
  data/                       # Recorded driving sessions (gitignored)
  models/                     # Saved model weights (gitignored)
```

## Usage

### Manual Driving (on Pi)

```bash
uv run python main.py drive
```

Opens a web UI for manual control and optional data recording.

### Training (on GPU machine)

```bash
uv run python -m src.training.pilotnet
```

Trains PilotNet on recorded data in `data/`. Saves the best model to `models/`.

### Autonomous Driving (on Pi)

```bash
uv run python main.py pilotnet
```

Loads the trained model and drives autonomously. Press Ctrl+C to stop.

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/). The virtual environment on the Pi uses `--system-site-packages` to access system-installed packages (`lgpio`, `picamera2`).

Key Python packages: PyTorch, torchvision, Tornado, Pillow, tqdm

## Future Improvements

- Horizontal flip augmentation to double training data and balance left/right turns
- Predict throttle in addition to steering
- Train on more diverse tracks and lighting conditions
- Add real-time telemetry overlay during autonomous driving
