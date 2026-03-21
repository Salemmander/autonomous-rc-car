# Training Notes

## Current Setup

- Architecture: PilotNet (NVIDIA end-to-end)
- Input: 120x160 RGB, top 30% cropped, YCbCr color space
- Output: steering only (throttle hardcoded at 0.2)
- Loss: MSE with Adam optimizer

## Roadmap

### Near-term

- **Recovery training**: collect data driving slightly off-track and correcting back
  - Pure data collection strategy, no code changes needed
- **Throttle output**: expand model to predict steering + throttle (2 outputs)
  - Requires logging throttle in training data

### Medium-term

- **Custom architecture**: experiment beyond PilotNet (attention, different backbones)
- **Stop sign detection**: separate detector or end-to-end with temporal context
  - Simple approach: classifier + control logic pause (no sequence model needed)
  - Advanced: frame sequence input so model learns "already stopped, can go now"

### Long-term

- **RL for recovery**: learn centering behavior via reward signal instead of imitation
- **Track mapping + navigation**: localization, path planning, directed waypoint driving
