import torch
from torch.utils.data import Dataset


class EmbeddedDataset(Dataset):
    """Dataset for sequence models with symbol embeddings.

    Expects a 3D tensor of shape (num_samples, seq_len, num_columns) where
    columns follow the layout: [date_id, time_id, symbol_id, weight, feature_0..feature_78, responder_6].
    """

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        features = self.data[idx, :, 4:83]
        target = self.data[idx, -1, 83]
        weight = self.data[idx, -1, 3]
        symbol_ids = self.data[idx, :, 2].to(torch.int)
        return features, target, weight, symbol_ids


class EmbeddedMLPDataset(Dataset):
    """Dataset for MLP models with symbol embeddings (no sequence dimension)."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        features = self.data[idx, 4:83]
        target = self.data[idx, 83]
        weight = self.data[idx, 3]
        symbol_id = self.data[idx, 2].to(torch.int)
        return features, target, weight, symbol_id


class FullDataset(Dataset):
    """Dataset for sequence models without symbol embeddings."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        features = self.data[idx, :, 4:83]
        target = self.data[idx, -1, 83]
        weight = self.data[idx, -1, 3]
        symbol_ids = self.data[idx, :, 2].to(torch.int)
        return features, target, weight, symbol_ids


class FullDatasetMLP(Dataset):
    """Dataset for MLP models without symbol embeddings (no sequence dimension)."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        features = self.data[idx, 4:83]
        target = self.data[idx, 83]
        weight = self.data[idx, 3]
        return features, target, weight
