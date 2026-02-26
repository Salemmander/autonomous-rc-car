from .dataset import DrivingDataset
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import os
from datetime import datetime


class Trainer:
    def __init__(
        self, model, data_path, transform, epochs, batch_size, lr, val_split
    ) -> None:
        self.model = model
        self.data_path = data_path
        self.transform = transform
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_split = val_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        dataset = DrivingDataset(self.data_path, self.transform)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        self.train_loader= DataLoader(train_set, batch_size=self.batch_size, shuffle=True)  # fmt:skip
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name = f"{self.model.__class__.__name__.lower()}_{date}.pth"

    def train(self, output="models/"):
        os.makedirs(output, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for images, steering in self.train_loader:
                images = images.to(self.device)
                steering = steering.to(self.device, dtype=torch.float32).unsqueeze(1)
                self.optimizer.zero_grad()
                pred = self.model(images)
                loss = self.criterion(pred, steering)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, steering in self.val_loader:
                    images = images.to(self.device)
                    steering = steering.to(self.device, dtype=torch.float32).unsqueeze(1)  # fmt:skip
                    pred = self.model(images)
                    loss = self.criterion(pred, steering)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)

            print(
                f"Epoch {epoch + 1}/{self.epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    self.model.state_dict(), os.path.join(output, self.model_name)
                )
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
