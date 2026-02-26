"""
Dataset for loading driving session data.
"""

from torch.utils.data import Dataset
import os
import glob
import csv
from PIL import Image


class DrivingDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.samples = []
        sessions = glob.glob(os.path.join(data_path, "session_*"))

        for session in sessions:
            with open(os.path.join(session, "log.csv")) as f:
                reader = csv.reader(f)
                next(reader)  # skip header row
                for row in reader:
                    image_name, _, steering, _ = row
                    full_path = os.path.join(session, image_name)
                    self.samples.append((full_path, float(steering)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, steering = self.samples[idx]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        return image, steering
