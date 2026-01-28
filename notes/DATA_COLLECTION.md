# Data Collection Plan

Guide for collecting training data for behavioral cloning.

## How It Works

Record `(image, steering, throttle)` tuples while driving. Your inputs ARE the labels - no manual labeling needed. The model learns to mimic your driving behavior.

## Recording Checklist

### Basic Coverage
- [ ] 10+ laps clockwise
- [ ] 10+ laps counter-clockwise
- [ ] Varied speeds (don't always drive at same throttle)

### Recovery Data
- [ ] Position car off-center, correct back to center
- [ ] Start at bad angles, recover
- [ ] This teaches the model how to fix mistakes

### Lighting Variations
- [ ] Daytime
- [ ] Evening / low light
- [ ] Overhead lights on/off
- [ ] Different shadow patterns

## Data Quality Tips

1. **Drive smoothly** - Jerky driving = jerky model (use a gamepad for analog control)
2. **Delete bad segments** - If you crash or mess up, remove that data
3. **Both directions** - Critical for balanced steering distribution
4. **Recovery matters** - A model trained only on perfect driving won't know how to recover

## Hardware Setup

### Camera
- Secure the camera mount - vibration causes blurry frames
- Check angle captures what the model needs (track edges, not ceiling)
- Test image quality before long recording sessions

### Track
- High contrast boundaries help (tape on floor, colored edges)
- Consistent setup between data collection and deployment
- Remove distractions/clutter from the track area if possible

### Battery
- Car behavior changes as battery drains (slower response)
- Note battery level in session log
- Consider keeping battery consistently charged during collection

## Before Training: Inspect Your Data

1. **Watch recordings back** - Play as video, spot obvious issues
2. **Check sync** - Verify steering/throttle values match what's happening visually
3. **Histogram steering angles** - Make sure you have a good distribution, not 90% zeros
4. **Check for imbalance** - Roughly equal left/right turns

## Frame Rate Considerations

- 20 FPS may be overkill - consecutive frames are nearly identical
- 10 FPS is likely sufficient for training
- More frames = more storage, longer training, diminishing returns

## Expected Dataset Size

- Simple track: 5-10k frames (10-20 laps at 20 FPS)
- More complex environment: 20-50k frames
- Start small, add more if model struggles

## Data Augmentation (Applied During Training)

These multiply your effective dataset size:

- **Horizontal flip** - Flip image and negate steering
- **Brightness jitter** - Random brightness/contrast changes
- **Small translations** - Shift image slightly, adjust steering

## Recording Format

Each recording session saves:
```
data/
  session_YYYYMMDD_HHMMSS/
    frame_00001.jpg
    frame_00002.jpg
    ...
    log.csv  (timestamp, steering, throttle)
```

## Session Log Template

| Date | Session | Laps | Direction | Lighting | Battery | Notes |
|------|---------|------|-----------|----------|---------|-------|
| | | | CW / CCW | | | |
| | | | CW / CCW | | | |
| | | | CW / CCW | | | |
