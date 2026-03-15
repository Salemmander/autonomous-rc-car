# Training Experiments

## Run 2: PilotNet + horizontal flip augmentation (2026-03-15)

- Model: `pilotnet_2026-03-15_19-58-13.pth`
- Dataset: 20,154 samples (10,077 originals + 10,077 horizontal flips with negated steering)
- Architecture: PilotNet (unchanged)
- Best val loss: **0.0512** at epoch 17 (val MAE: 0.1735)
- Early stopped at epoch 27
- Overfitting visible: train loss kept dropping (0.031) while val loss rose (0.057)

## Run 1: PilotNet baseline (2026-02-28)

- Model: `pilotnet_2026-02-28_12-03-31.pth`
- Dataset: 10,077 samples (no augmentation)
- Architecture: PilotNet
- Best val loss: **0.0509** at epoch 18 (val MAE: 0.1738)
- Early stopped at epoch 28
- Same overfitting pattern as run 2

## Comparison

| Metric        | Run 1 (baseline) | Run 2 (flip aug) |
| ------------- | ---------------- | ---------------- |
| Dataset size  | 10,077           | 20,154           |
| Best val loss | 0.0509           | 0.0512           |
| Best val MAE  | 0.1738           | 0.1735           |
| Best epoch    | 18               | 17               |
| Early stop    | 28               | 27               |

Val performance is essentially identical. The flip augmentation didn't hurt, but didn't meaningfully improve validation metrics either. This suggests the track data is already fairly symmetric, or the model needs more diverse data (different positions, recovery maneuvers) rather than more of the same.
