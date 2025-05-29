import torch


def sigmoid_loss(y: torch.Tensor, logits: torch.Tensor, temperature: float = 1.0):
    return torch.where(
        y == 1,
        1 / (1 + torch.exp(logits * temperature)),
        1 / (1 + torch.exp(-logits * temperature)),
    ).mean()


def logistic_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(
        y == 1, torch.log(1 + torch.exp(-logits)), torch.log(1 + torch.exp(logits))
    ).mean()


def hinge_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(
        y == 1, torch.clamp(1 - logits, min=0), torch.clamp(1 + logits, min=0)
    ).mean()


def square_loss(y: torch.Tensor, logits: torch.Tensor):
    # return ((y - logits) ** 2).mean()
    return torch.where(y == 1, (logits - 1) ** 2, (logits + 1) ** 2).mean()


def ramp_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(
        y == 1,
        torch.clamp(1 - logits, min=0, max=1),
        torch.clamp(1 + logits, min=0, max=1),
    ).mean()


def exp_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(y == 1, torch.exp(-logits), torch.exp(logits)).mean()


def unhinged_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(y == 1, 1 - logits, 1 + logits).mean()


def symmetric_ramp_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(
        y == 1,
        torch.clamp(0.5 - 0.5 * logits, min=0, max=1),
        torch.clamp(0.5 + 0.5 * logits, min=0, max=1),
    ).mean()
