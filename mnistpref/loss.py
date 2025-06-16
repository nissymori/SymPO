from functools import partial
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


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


def symmetric_ramp_loss(y: torch.Tensor, logits: torch.Tensor, slope: float = 0.5):
    return torch.where(
        y == 1,
        torch.clamp(0.5 - slope * logits, min=0, max=1),
        torch.clamp(0.5 + slope * logits, min=0, max=1),
    ).mean()


def exp_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(y == 1, torch.exp(-logits), torch.exp(logits)).mean()


def unhinged_loss(y: torch.Tensor, logits: torch.Tensor):
    return torch.where(y == 1, 1 - logits, 1 + logits).mean()


def make_loss_fn(loss_type: str, config: Dict[str, Any]):
    if loss_type == "sigmoid":
        return partial(sigmoid_loss, temperature=config.sigmoid_temperature)
    elif loss_type == "logistic":
        return logistic_loss
    elif loss_type == "hinge":
        return hinge_loss
    elif loss_type == "square":
        return square_loss
    elif loss_type == "ramp":
        return ramp_loss
    elif loss_type == "symmetric_ramp":
        return partial(symmetric_ramp_loss, slope=config.slope)
    elif loss_type == "exp":
        return exp_loss
    elif loss_type == "unhinged":
        return unhinged_loss
    else:
        raise ValueError("Unknown loss type.")


def move_batch_to_device(
    batch: Tuple[torch.Tensor, ...], device: torch.device
) -> Tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in batch)


def sympo_loss(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Use the noisy label as the target.
    """
    loss_fn = make_loss_fn(config.loss_type, config)
    x, y, y_true, r_diff, _ = batch
    r_1 = model(x[:, 0, :, :].unsqueeze(1)).squeeze()
    r_2 = model(x[:, 1, :, :].unsqueeze(1)).squeeze()
    logits = r_1 - r_2
    loss = loss_fn(y, logits)
    return loss


def pn_loss(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Use the original label as the target.
    """
    loss_fn = make_loss_fn(config.loss_type, config)
    x, y, y_true, r_diff, _ = batch
    r_1 = model(x[:, 0, :, :].unsqueeze(1)).squeeze()
    r_2 = model(x[:, 1, :, :].unsqueeze(1)).squeeze()
    logits = r_1 - r_2
    loss = loss_fn(y_true, logits)
    return loss


def calc_metrics(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[float, float, float]:
    model.eval()
    # Move batch to device
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        # Extract batch elements
        x, y, y_true, r_diff, _ = batch
        r_label = r_diff > 0
        r_1 = model(x[:, 0, :, :].unsqueeze(1)).squeeze()
        r_2 = model(x[:, 1, :, :].unsqueeze(1)).squeeze()
        logits = r_1 - r_2
        noisy_label_is_correct = y == y_true
        noisy_label_is_incorrect = y != y_true
        avg_reward_abs = (r_1.abs() + r_2.abs()) / 2
        avg_logit_abs = logits.abs()

        avg_reward_abs_correct = (
            (r_1[noisy_label_is_correct].abs() + r_2[noisy_label_is_correct].abs()) / 2
        ).mean()
        avg_reward_abs_incorrect = (
            (r_1[noisy_label_is_incorrect].abs() + r_2[noisy_label_is_incorrect].abs()) / 2
        ).mean()
        avg_logit_abs_correct = (logits[noisy_label_is_correct].abs()).mean()
        avg_logit_abs_incorrect = (logits[noisy_label_is_incorrect].abs()).mean()
        avg_reward_abs = avg_reward_abs.mean()
        avg_logit_abs = avg_logit_abs.mean()

        pred = logits > 0.0
        noisy_label_correct = (pred.reshape(-1) == y.reshape(-1)).sum().item()
        true_label_correct = (pred.reshape(-1) == y_true.reshape(-1)).sum().item()
        reward_correct = (pred.reshape(-1) == r_label.reshape(-1)).sum().item()

        noisy_label_correct_for_correct = (
            (
                pred[noisy_label_is_correct].reshape(-1)
                == y[noisy_label_is_correct].reshape(-1)
            )
            .sum()
            .item()
        )
        noisy_label_correct_for_incorrect = (
            (
                pred[noisy_label_is_incorrect].reshape(-1)
                == y[noisy_label_is_incorrect].reshape(-1)
            )
            .sum()
            .item()
        )
    return {
        "noisy_label_correct": noisy_label_correct / len(y),
        "noisy_label_correct_for_correct": noisy_label_correct_for_correct
        / len(y[noisy_label_is_correct]),
        "noisy_label_correct_for_incorrect": noisy_label_correct_for_incorrect
        / len(y[noisy_label_is_incorrect]),
        "true_label_correct": true_label_correct / len(y_true),
        "reward_correct": reward_correct / len(r_diff),
        "avg_reward_abs": avg_reward_abs,
        "avg_reward_abs_correct": avg_reward_abs_correct,
        "avg_reward_abs_incorrect": avg_reward_abs_incorrect,
        "avg_logit_abs": avg_logit_abs,
        "avg_logit_abs_correct": avg_logit_abs_correct,
        "avg_logit_abs_incorrect": avg_logit_abs_incorrect,
    }


def metrics_keys():
    return [
        "noisy_label_correct",
        "noisy_label_correct_for_correct",
        "noisy_label_correct_for_incorrect",
        "true_label_correct",
        "reward_correct",
        "avg_reward_abs",
        "avg_reward_abs_correct",
        "avg_reward_abs_incorrect",
        "avg_logit_abs",
        "avg_logit_abs_correct",
        "avg_logit_abs_incorrect",
    ]


def compute_gradient_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> nn.Module:
    model.to(device)
    train_metrics = {key: 0.0 for key in metrics_keys()}
    train_loss = 0.0
    total = 0
    for epoch in range(config.num_epochs):
        for batch_idx, (batch) in enumerate(train_loader):
            model.train()
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            if config.algo == "sympo":
                loss = sympo_loss(model, batch, device, config)
            elif config.algo == "pn":
                loss = pn_loss(model, batch, device, config)
            else:
                raise ValueError("Unknown algorithm.")
            loss.backward()
            optimizer.step()
            metrics = calc_metrics(model, batch, device, config)
            for key in metrics_keys():  # update train_metrics
                train_metrics[key] += metrics[key]
            train_loss += loss.item()
            total += 1
        for key in metrics_keys():  # average train_metrics
            train_metrics[key] /= total
        train_loss /= total
        if epoch % 10 == 0:
            test_metrics = test(model, test_loader, device, config, epoch)
            print(
                f"""
                Epoch: {epoch}, \\
                Train Label Acc: {train_metrics["noisy_label_correct"] * 100:.2f}, \\
                Train Noisy Label Acc: {train_metrics["noisy_label_correct"] * 100:.2f}, \\
                Train Noisy Label Acc for Correct: {train_metrics["noisy_label_correct_for_correct"] * 100:.2f}, \\
                Train Noisy Label Acc for Incorrect: {train_metrics["noisy_label_correct_for_incorrect"] * 100:.2f}, \\
                Test Label Accuracy: {test_metrics["noisy_label_correct"] * 100:.2f}, \\
                Test Noisy Label Acc: {test_metrics["noisy_label_correct"] * 100:.2f}, \\
                Test Reward Acc: {test_metrics["reward_correct"] * 100:.2f}, \\
                avg_logit_abs: {test_metrics["avg_logit_abs"] * 100:.2f}, \\
                train_loss: {train_loss:.2f}
                """
            )
            # train_のprefixを追加
            log_train_metrics = {}
            for key in train_metrics.keys():  # add train_ prefix
                log_train_metrics[f"train_{key}"] = train_metrics[key]
            log_test_metrics = {}
            for key in test_metrics.keys():  # add test_ prefix
                log_test_metrics[f"test_{key}"] = test_metrics[key]
            wandb.log(
                log_train_metrics
                | log_test_metrics
                | {
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
            )
    return model


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    epoch: int,
) -> Tuple[float, float, float]:
    model.eval()
    test_metrics = {key: 0.0 for key in metrics_keys()}
    total = 0
    for batch in test_loader:
        metrics = calc_metrics(model, batch, device, config)
        for key in metrics_keys():  # update test_metrics
            test_metrics[key] += metrics[key]
        total += 1
    for key in metrics_keys():  # average test_metrics
        test_metrics[key] /= total
    return test_metrics


def final_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        metrics = {key: 0.0 for key in metrics_keys()}
        reward_diffs = [[] for _ in range(19)]
        rewards = [[] for _ in range(19)]
        total = 0
        for batch in test_loader:
            metrics = calc_metrics(model, batch, device, config)
            for key in metrics_keys():  # update metrics
                metrics[key] += metrics[key]
            total += 1
            x, y, y_true, reward_diff, true_reward = move_batch_to_device(batch, device)
            r_1 = model(x[:, 0, :, :].unsqueeze(1)).squeeze()
            r_2 = model(x[:, 1, :, :].unsqueeze(1)).squeeze()
            logits = r_1 - r_2
            for i in range(len(reward_diff)):
                diff = int(reward_diff[i].item())
                idx = diff + 9
                reward_diffs[idx].append(logits[i].item())
            for i in range(len(true_reward)):
                idx1, idx2 = int(true_reward[i][0].item()), int(
                    true_reward[i][1].item()
                )
                rewards[idx1].append(r_1[i].item())
                rewards[idx2].append(r_2[i].item())
    return metrics, reward_diffs, rewards
