# Training Notes

## Future Improvements

- Horizontal flip: flip image left-right and negate steering angle to double the dataset
  - Do this in DrivingDataset.**getitem** since it needs access to both image and steering
  - PIL: `img.transpose(Image.FLIP_LEFT_RIGHT)`, then `steering = -steering`
  - Pre-duplicate in **init** so every sample has a flipped copy in self.samples
