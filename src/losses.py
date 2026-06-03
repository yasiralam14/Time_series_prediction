import torch
import torch.nn as nn


def weighted_mse(
    predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    loss_fn = nn.MSELoss(reduction="none")
    mse_loss = loss_fn(predictions, targets)
    weighted_loss = mse_loss * weights
    return weighted_loss.mean()


def weighted_mae(
    targets: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    absolute_errors = torch.abs(targets - predictions)
    weighted_errors = absolute_errors * weights
    return torch.sum(weighted_errors) / torch.sum(weights)


def mae(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(targets - predictions))


def huber_loss(
    targets: torch.Tensor, predictions: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    errors = torch.abs(targets - predictions)
    is_small = errors <= delta
    small_loss = 0.5 * (errors[is_small] ** 2)
    large_loss = delta * (errors[~is_small] - 0.5 * delta)
    return (torch.sum(small_loss) + torch.sum(large_loss)) / len(targets)


def weighted_huber_loss(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    errors = torch.abs(targets - predictions)
    is_small = errors <= delta
    small_loss = 0.5 * (errors[is_small] ** 2) * weights[is_small]
    large_loss = delta * (errors[~is_small] - 0.5 * delta) * weights[~is_small]
    return (torch.sum(small_loss) + torch.sum(large_loss)) / torch.sum(weights)
