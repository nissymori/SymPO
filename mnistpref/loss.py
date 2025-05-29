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


def make_logits(
    model: nn.Module,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    output_reward: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    img_11 = x_1[:, 0, :, :]
    img_12 = x_1[:, 1, :, :]
    img_21 = x_2[:, 0, :, :]
    img_22 = x_2[:, 1, :, :]
    r_11 = model(img_11.unsqueeze(1)).squeeze()
    r_12 = model(img_12.unsqueeze(1)).squeeze()
    r_21 = model(img_21.unsqueeze(1)).squeeze()
    r_22 = model(img_22.unsqueeze(1)).squeeze()
    logits1 = r_11 - r_12
    logits2 = r_21 - r_22
    if output_reward:
        return logits1, logits2, r_11, r_12, r_21, r_22
    else:
        return logits1, logits2


def sympo_loss(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    config: Dict[str, Any],
) -> torch.Tensor:
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
    loss_fn = make_loss_fn(config.loss_type, config)
    x, y, y_true, r_diff, _ = batch
    r_1 = model(x[:, 0, :, :].unsqueeze(1)).squeeze()
    r_2 = model(x[:, 1, :, :].unsqueeze(1)).squeeze()
    logits = r_1 - r_2
    loss = loss_fn(y_true, logits)
    return loss


def nll_loss(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    config: Dict[str, Any],
) -> torch.Tensor:
    loss_fn = make_loss_fn(config.loss_type, config)
    x, y, y_true, r_diff, _ = batch
    r_1 = model(x[:, 0, :, :].unsqueeze(1)).squeeze()
    r_2 = model(x[:, 1, :, :].unsqueeze(1)).squeeze()
    logits = r_1 - r_2
    original_loss = loss_fn(y, logits)
    flipped_loss = loss_fn(1 - y, logits)
    loss = (
        original_loss * (1 - config.epsilon_n) + flipped_loss * config.epsilon_p
    ) / (1 - config.epsilon_p - config.epsilon_n)
    return loss


def calc_accuracy(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[float, float, float]:
    model.eval()
    # Move batch to device
    batch = move_batch_to_device(batch, device)
    model = model.to(device)
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
    return (
        noisy_label_correct / len(y),
        noisy_label_correct_for_correct / len(y[noisy_label_is_correct]),
        noisy_label_correct_for_incorrect / len(y[noisy_label_is_incorrect]),
        true_label_correct / len(y_true),
        reward_correct / len(r_diff),
        avg_reward_abs,
        avg_reward_abs_correct,
        avg_reward_abs_incorrect,
        avg_logit_abs,
        avg_logit_abs_correct,
        avg_logit_abs_incorrect,
    )


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
    for epoch in range(config.num_epochs):
        train_label_acc = 0
        train_noisy_label_acc = 0
        train_noisy_label_acc_for_correct = 0
        train_noisy_label_acc_for_incorrect = 0
        train_reward_acc = 0
        train_loss = 0
        train_avg_reward_abs = 0
        train_avg_reward_abs_correct = 0
        train_avg_reward_abs_incorrect = 0
        train_avg_logit_abs = 0
        train_avg_logit_abs_correct = 0
        train_avg_logit_abs_incorrect = 0
        train_gradient_norm = 0
        total = 0
        for batch_idx, (batch) in enumerate(train_loader):
            model.train()
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            if config.algo == "sympo":
                loss = sympo_loss(model, batch, device, config)
            elif config.algo == "nll":
                loss = nll_loss(model, batch, device, config)
            elif config.algo == "pn":
                loss = pn_loss(model, batch, device, config)
            else:
                raise ValueError("Unknown algorithm.")
            loss.backward()
            train_gradient_norm += compute_gradient_norm(model)
            optimizer.step()
            (
                noisy_label_acc,
                noisy_label_acc_for_correct,
                noisy_label_acc_for_incorrect,
                label_acc,
                reward_acc,
                avg_reward_abs,
                avg_reward_abs_correct,
                avg_reward_abs_incorrect,
                avg_logit_abs,
                avg_logit_abs_correct,
                avg_logit_abs_incorrect,
            ) = calc_accuracy(model, batch, device, config)
            train_label_acc += label_acc
            train_noisy_label_acc += noisy_label_acc
            train_noisy_label_acc_for_correct += noisy_label_acc_for_correct
            train_noisy_label_acc_for_incorrect += noisy_label_acc_for_incorrect
            train_reward_acc += reward_acc
            train_avg_reward_abs += avg_reward_abs
            train_avg_reward_abs_correct += avg_reward_abs_correct
            train_avg_reward_abs_incorrect += avg_reward_abs_incorrect
            train_avg_logit_abs += avg_logit_abs
            train_avg_logit_abs_correct += avg_logit_abs_correct
            train_avg_logit_abs_incorrect += avg_logit_abs_incorrect
            train_loss += loss.item()
            total += 1
        train_label_acc /= total
        train_noisy_label_acc /= total
        train_noisy_label_acc_for_correct /= total
        train_noisy_label_acc_for_incorrect /= total
        train_reward_acc /= total
        train_loss /= total
        train_avg_reward_abs /= total
        train_avg_reward_abs_correct /= total
        train_avg_reward_abs_incorrect /= total
        train_avg_logit_abs /= total
        train_avg_logit_abs_correct /= total
        train_avg_logit_abs_incorrect /= total
        train_gradient_norm /= total
        if epoch % 10 == 0:
            tsne(model, train_loader, device, config, epoch)
            (
                noisy_label_acc,
                noisy_label_acc_for_correct,
                noisy_label_acc_for_incorrect,
                label_acc,
                reward_acc,
                avg_reward_abs,
                avg_reward_abs_correct,
                avg_reward_abs_incorrect,
                avg_logit_abs,
                avg_logit_abs_correct,
                avg_logit_abs_incorrect,
            ) = test(model, test_loader, device, config, epoch)
            print(
                f"""
                Epoch: {epoch}, \\
                Train Label Acc: {train_label_acc * 100:.2f}, \\
                Train Noisy Label Acc: {train_noisy_label_acc * 100:.2f}, \\
                Train Noisy Label Acc for Correct: {train_noisy_label_acc_for_correct * 100:.2f}, \\
                Train Noisy Label Acc for Incorrect: {train_noisy_label_acc_for_incorrect * 100:.2f}, \\
                Test Label Accuracy: {label_acc * 100:.2f}, \\
                Test Noisy Label Acc: {noisy_label_acc * 100:.2f}, \\
                Test Reward Acc: {reward_acc * 100:.2f}, \\
                avg_logit_abs: {avg_logit_abs:.2f}, \\
                train_gradient_norm: {train_gradient_norm:.2f}
                """
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train_label_acc": train_label_acc,
                    "train_noisy_label_acc": train_noisy_label_acc,
                    "train_avg_reward_abs": train_avg_reward_abs,
                    "train_avg_reward_abs_correct": train_avg_reward_abs_correct,
                    "train_avg_reward_abs_incorrect": train_avg_reward_abs_incorrect,
                    "train_avg_logit_abs": train_avg_logit_abs,
                    "train_avg_logit_abs_correct": train_avg_logit_abs_correct,
                    "train_avg_logit_abs_incorrect": train_avg_logit_abs_incorrect,
                    "test_label_acc": label_acc,
                    "test_noisy_label_acc": noisy_label_acc,
                    "test_noisy_label_acc_for_correct": noisy_label_acc_for_correct,
                    "test_noisy_label_acc_for_incorrect": noisy_label_acc_for_incorrect,
                    "test_reward_acc": reward_acc,
                    "test_avg_reward_abs": avg_reward_abs,
                    "test_avg_reward_abs_correct": avg_reward_abs_correct,
                    "test_avg_reward_abs_incorrect": avg_reward_abs_incorrect,
                    "test_avg_logit_abs": avg_logit_abs,
                    "test_avg_logit_abs_correct": avg_logit_abs_correct,
                    "test_avg_logit_abs_incorrect": avg_logit_abs_incorrect,
                    "train_gradient_norm": train_gradient_norm,
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
    noisy_label_acc = 0
    noisy_label_acc_for_correct = 0
    noisy_label_acc_for_incorrect = 0
    label_acc = 0
    reward_acc = 0
    avg_reward_abs = 0
    avg_reward_abs_correct = 0
    avg_reward_abs_incorrect = 0
    avg_logit_abs = 0
    avg_logit_abs_correct = 0
    avg_logit_abs_incorrect = 0
    total = 0
    for batch in test_loader:
        (
            n_acc,
            n_acc_for_correct,
            n_acc_for_incorrect,
            t_acc,
            r_acc,
            reward_abs,
            reward_abs_correct,
            reward_abs_incorrect,
            logit_abs,
            logit_abs_correct,
            logit_abs_incorrect,
        ) = calc_accuracy(model, batch, device, config)
        noisy_label_acc += n_acc
        noisy_label_acc_for_correct += n_acc_for_correct
        noisy_label_acc_for_incorrect += n_acc_for_incorrect
        label_acc += t_acc
        reward_acc += r_acc
        avg_reward_abs += reward_abs
        avg_reward_abs_correct += reward_abs_correct
        avg_reward_abs_incorrect += reward_abs_incorrect
        avg_logit_abs += logit_abs
        avg_logit_abs_correct += logit_abs_correct
        avg_logit_abs_incorrect += logit_abs_incorrect
        total += 1
    avg_reward_abs /= total
    avg_reward_abs_correct /= total
    avg_reward_abs_incorrect /= total
    avg_logit_abs /= total
    avg_logit_abs_correct /= total
    avg_logit_abs_incorrect /= total
    noisy_label_acc /= total
    noisy_label_acc_for_correct /= total
    noisy_label_acc_for_incorrect /= total
    return (
        noisy_label_acc / total,
        noisy_label_acc_for_correct / total,
        noisy_label_acc_for_incorrect / total,
        label_acc / total,
        reward_acc / total,
        avg_reward_abs,
        avg_reward_abs_correct,
        avg_reward_abs_incorrect,
        avg_logit_abs,
        avg_logit_abs_correct,
        avg_logit_abs_incorrect,
    )


def final_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        noisy_label_acc = 0
        noisy_label_acc_for_correct = 0
        noisy_label_acc_for_incorrect = 0
        true_label_acc = 0
        reward_acc = 0
        reward_diffs = [[] for _ in range(19)]
        rewards = [[] for _ in range(19)]
        total = 0
        for batch in test_loader:
            (
                n_acc,
                n_acc_for_correct,
                n_acc_for_incorrect,
                t_acc,
                r_acc,
                reward_abs,
                reward_abs_correct,
                reward_abs_incorrect,
                logit_abs,
                logit_abs_correct,
                logit_abs_incorrect,
            ) = calc_accuracy(model, batch, device, config)
            noisy_label_acc += n_acc
            noisy_label_acc_for_correct += n_acc_for_correct
            noisy_label_acc_for_incorrect += n_acc_for_incorrect
            true_label_acc += t_acc
            reward_acc += r_acc
            total += 1
            (
                x,
                y,
                y_true,
                reward_diff,
                true_reward,
            ) = move_batch_to_device(batch, device)
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
    return (
        noisy_label_acc / total,
        noisy_label_acc_for_correct / total,
        noisy_label_acc_for_incorrect / total,
        true_label_acc / total,
        reward_acc / total,
        reward_diffs,
        rewards,
    )


def tsne(model, dataloader, device, config, epoch):
    model.eval()
    with torch.no_grad():
        all_x = []
        all_noised_Y = []
        all_true_Y = []
        all_Y_diff = []
        all_true_rewards = []
        idx = 0
        # dataloーダーから各バッチのデータを抽出
        for batch in dataloader:
            x, noised_Y, true_Y, Y_diff, true_rewards = batch
            x = x.to(device)
            model = model.to(device)
            # 各ペア画像でforwardし、出力の差分を算出
            _, output_1 = model(x[:, 0, :, :].unsqueeze(1), visualize=True)
            _, output_2 = model(x[:, 1, :, :].unsqueeze(1), visualize=True)
            output = output_1.squeeze() - output_2.squeeze()

            all_x.append(output)
            all_noised_Y.append(noised_Y)
            all_true_Y.append(true_Y)
            all_Y_diff.append(Y_diff)
            all_true_rewards.append(true_rewards)
            idx += 1
            if idx > 5:
                break

        # 各バッチのリストをテンソルとして連結
        all_x = torch.cat(all_x, dim=0)
        all_noised_Y = torch.cat(all_noised_Y, dim=0)
        all_true_Y = torch.cat(all_true_Y, dim=0)
        all_Y_diff = torch.cat(all_Y_diff, dim=0)
        all_true_rewards = torch.cat(all_true_rewards, dim=0)

        # 正解・不正解のマスクを作成（noised_Yとtrue_Yが一致するか否か）
        correct_mask = all_noised_Y == all_true_Y
        incorrect_mask = all_noised_Y != all_true_Y

        # trueラベルがPositive（1）かNegative（0）のマスクを作成
        positive_mask = all_true_Y == 1
        negative_mask = all_true_Y == 0

        # 各条件でインデックスを抽出
        correct_positive_idx = (correct_mask & positive_mask).nonzero().squeeze()
        correct_negative_idx = (correct_mask & negative_mask).nonzero().squeeze()
        incorrect_positive_idx = (incorrect_mask & positive_mask).nonzero().squeeze()
        incorrect_negative_idx = (incorrect_mask & negative_mask).nonzero().squeeze()

        # TSNEで全データを2次元に圧縮
        from sklearn.manifold import TSNE

        tsne_obj = TSNE(n_components=2, random_state=42)
        emb = tsne_obj.fit_transform(all_x.cpu().numpy())

        import matplotlib.pyplot as plt

        # 2つの図を左右に並べたsubplotとして表示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左側: 正解のデータをプロット (正: 青, 負: 赤)
        ax1.scatter(
            emb[correct_positive_idx, 0],
            emb[correct_positive_idx, 1],
            c="blue",
            marker="o",
            label="Correct Positive",
        )
        ax1.scatter(
            emb[correct_negative_idx, 0],
            emb[correct_negative_idx, 1],
            c="red",
            marker="o",
            label="Correct Negative",
        )
        ax1.set_title("t-SNE: Correct Predictions (Epoch {})".format(epoch))
        ax1.legend()

        # 右側: 誤分類のデータをプロット (正: 青, 負: 赤)
        ax2.scatter(
            emb[incorrect_positive_idx, 0],
            emb[incorrect_positive_idx, 1],
            c="blue",
            marker="o",
            label="Incorrect Positive",
        )
        ax2.scatter(
            emb[incorrect_negative_idx, 0],
            emb[incorrect_negative_idx, 1],
            c="red",
            marker="o",
            label="Incorrect Negative",
        )
        ax2.set_title("t-SNE: Incorrect Predictions (Epoch {})".format(epoch))
        ax2.legend()

        # 画像として保存（例：tsne_combined_epoch.png）
        plt.savefig(f"fig/tsne_{config.loss_type}_{config.epsilon_p}_{epoch}.png")
        plt.close()
