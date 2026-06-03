import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class EmbeddedMLP(nn.Module):
    def __init__(self, num_features: int, num_symbols: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.fc1 = nn.Linear(num_features + embedding_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor, symbol_id: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(symbol_id)
        x = torch.cat([features, embedded], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class ELSTM(nn.Module):
    """LSTM with learned symbol embeddings concatenated to input features."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size + embedding_dim, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        symbol_embeds = self.embedding(symbol_ids)
        x = torch.cat((x, symbol_embeds), dim=-1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class FLSTM(nn.Module):
    """LSTM with a learned feature-engineering front-end and symbol embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        refined_features_size: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            refined_features_size + embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, refined_features_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        symbol_embeds = self.embedding(symbol_ids)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.cat((x, symbol_embeds), dim=-1)
        lstm_out, _ = self.lstm(x)
        return self.fc3(lstm_out[:, -1, :])
