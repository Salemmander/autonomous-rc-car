from .dataset import DrivingDataset
import csv
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import os
from datetime import datetime


class Trainer:
    def __init__(
        self,
        model,
        data_path,
        transform,
        max_epochs: int,
        patience: int,
        batch_size: int,
        lr: float,
        val_split: float,
    ) -> None:
        self.model = model
        self.data_path = data_path
        self.transform = transform
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.val_split = val_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model = self.model.to(self.device)
        dataset = DrivingDataset(self.data_path, self.transform)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        self.train_loader= DataLoader(train_set, batch_size=self.batch_size, shuffle=True)  # fmt:skip
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        self.criterion = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name = f"{self.model.__class__.__name__.lower()}_{date}.pth"
        self.metrics_name = self.model_name.replace(".pth", "_metrics.csv")

    def train(self, output="models/"):
        os.makedirs(output, exist_ok=True)
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        epoch = 0

        metrics_path = os.path.join(output, self.metrics_name)
        csv_file = open(metrics_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_mae", "val_mae"])

        while epoch < self.max_epochs and epochs_without_improvement < self.patience:
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            for images, steering in self.train_loader:
                images = images.to(self.device)
                steering = steering.to(self.device, dtype=torch.float32).unsqueeze(1)
                self.optimizer.zero_grad()
                pred = self.model(images)
                loss = self.criterion(pred, steering)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_mae += self.mae(pred, steering).item()

            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_mae = train_mae / len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            with torch.no_grad():
                for images, steering in self.val_loader:
                    images = images.to(self.device)
                    steering = steering.to(self.device, dtype=torch.float32).unsqueeze(1)  # fmt:skip
                    pred = self.model(images)
                    loss = self.criterion(pred, steering)
                    val_loss += loss.item()
                    val_mae += self.mae(pred, steering).item()

            avg_val_loss = val_loss / len(self.val_loader)
            avg_val_mae = val_mae / len(self.val_loader)

            print(
                f"Epoch {epoch + 1} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, train_mae: {avg_train_mae:.4f}, val_mae: {avg_val_mae:.4f}"
            )

            writer.writerow(
                [
                    epoch + 1,
                    f"{avg_train_loss:.4f}",
                    f"{avg_val_loss:.4f}",
                    f"{avg_train_mae:.4f}",
                    f"{avg_val_mae:.4f}",
                ]
            )
            csv_file.flush()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(
                    self.model.state_dict(), os.path.join(output, self.model_name)
                )
            else:
                epochs_without_improvement += 1
            epoch += 1

        if epochs_without_improvement >= self.patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)"
            )
        csv_file.close()
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
