from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.losses import mae, weighted_mse
from src.metrics import weighted_r2
from src.preprocessing import batch_by_timestamp, normalize


class Trainer:
    """Unified training and evaluation loop for all PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        loss_fn: Callable | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = loss_fn or mae

    def train_epoch(self, dataloader: DataLoader, use_symbols: bool = False) -> float:
        self.model.train()
        running_loss = 0.0
        for batch in dataloader:
            if use_symbols:
                inputs, targets, weights, symbols = batch
                symbols = symbols.to(self.device)
            else:
                inputs, targets, weights = batch[:3]
                symbols = None

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            weights = weights.to(self.device)

            self.optimizer.zero_grad()
            predictions = (
                self.model(inputs, symbols) if symbols is not None else self.model(inputs)
            )
            loss = self.loss_fn(predictions.squeeze(), targets.float())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        return running_loss / max(len(dataloader), 1)

    def fit(
        self, dataloader: DataLoader, epochs: int = 10, use_symbols: bool = False
    ) -> list[float]:
        losses = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, use_symbols=use_symbols)
            losses.append(avg_loss)
            print(f"  Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.6f}")
        return losses

    @torch.no_grad()
    def evaluate(
        self, dataloader: DataLoader, use_symbols: bool = False
    ) -> torch.Tensor:
        self.model.eval()
        all_preds = []
        for batch in dataloader:
            if use_symbols:
                inputs, _, _, symbols = batch
                symbols = symbols.to(self.device)
            else:
                inputs = batch[0]
                symbols = None

            inputs = inputs.to(self.device)
            outputs = (
                self.model(inputs, symbols) if symbols is not None else self.model(inputs)
            )
            all_preds.append(outputs.cpu())
        return torch.cat(all_preds, dim=0)

    def online_evaluate(
        self,
        data: torch.Tensor,
        sequence_size: int = 5,
        max_batches: int | None = None,
        use_symbols: bool = False,
    ) -> dict:
        """Evaluate with online (continual) learning on time-ordered batches."""
        data_normed = data.clone()
        data_normed[:, 4:83] = normalize(data_normed[:, 4:83])
        batches = batch_by_timestamp(data_normed)

        self.model.train()
        predictions, targets, weights = [], [], []
        limit = max_batches or (len(batches) - sequence_size + 1)

        for idx in range(min(limit, len(batches) - sequence_size + 1)):
            seq = batches[idx : idx + sequence_size]
            seq_tensor = torch.stack(seq).permute(1, 0, 2)

            feat = seq_tensor[:, :, 4:83].to(self.device)
            y = seq_tensor[:, -1, 83].to(self.device)
            w = seq_tensor[:, -1, 3].to(self.device)

            if use_symbols:
                sym = seq_tensor[:, :, 2].to(torch.int).to(self.device)
                outputs = self.model(feat, sym)
            else:
                outputs = self.model(feat)

            self.optimizer.zero_grad()
            loss = weighted_mse(outputs.squeeze(), y.float(), w.float())
            loss.backward()
            self.optimizer.step()

            predictions.append(outputs.squeeze().detach().cpu())
            targets.append(y.squeeze().cpu())
            weights.append(w.squeeze().cpu())

        preds = torch.cat(predictions)
        tgts = torch.cat(targets)
        wts = torch.cat(weights)
        r2 = weighted_r2(preds, tgts, wts)

        return {"predictions": preds, "targets": tgts, "weights": wts, "weighted_r2": r2.item()}
