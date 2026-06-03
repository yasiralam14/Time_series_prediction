import torch


def weighted_r2(
    predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    assert torch.all(weights >= 0), "Weights must be non-negative."

    weighted_mean = torch.sum(weights * targets) / torch.sum(weights)
    ss_total = torch.sum(weights * (targets - weighted_mean) ** 2)
    ss_residual = torch.sum(weights * (targets - predictions) ** 2)

    if ss_total.item() == 0:
        return torch.tensor(1.0) if ss_residual.item() == 0 else torch.tensor(0.0)

    return 1 - (ss_residual / ss_total)
