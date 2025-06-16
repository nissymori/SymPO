import copy
from functools import partial
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


def noise_model(
    label: int,
    add_noise: bool,
    symmetric: bool,
    instance_dependent_noise: bool,
    max_noise_ratio: float,
    epsilon_p: float,
    epsilon_n: float,
):
    if add_noise:
        if symmetric:
            assert epsilon_p == epsilon_n
            if instance_dependent_noise:
                noise_ratio = np.random.uniform(0, max_noise_ratio)
                is_flipped = np.random.binomial(1, noise_ratio)
                noised_label = 1 - label if is_flipped else label
            else:
                is_flipped = np.random.binomial(1, epsilon_p)
                noised_label = 1 - label if is_flipped else label
        else:
            if instance_dependent_noise:
                noise_ratio = np.random.uniform(0, max_noise_ratio)
                is_flipped = np.random.binomial(1, noise_ratio)
                noised_label = 1 - label if is_flipped else label
            else:
                if label == 1:
                    is_flipped = np.random.binomial(1, epsilon_p)
                    noised_label = 1 - label if is_flipped else label
                else:
                    is_flipped = np.random.binomial(1, epsilon_n)
                    noised_label = 1 - label if is_flipped else label
    else:
        noised_label = label
    return noised_label


def make_paired_pref_data(
    X: np.ndarray,
    Y: np.ndarray,
    tempereture=1,
    digits_included=None,
    allow_par=True,
    add_noise=True,
    symmetric=True,
    instance_dependent_noise=False,
    max_noise_ratio=0.3,
    epsilon_p=0.3,
    epsilon_n=0.3,
    class_prior=0.5,  # Target class prior
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate paired MNIST data with class prior adjustment.

    Parameters:
        class_prior (float): Desired proportion of label 1 in the dataset.

    Returns:
        Tuple of paired input, noisy labels, true labels, differences, and true rewards.
    """
    data_num = len(X) // 2
    train_X_pair = []
    train_label = []
    true_label = []
    train_diff = []
    true_rewards = []

    noise_fn = partial(
        noise_model,
        add_noise=add_noise,
        symmetric=symmetric,
        instance_dependent_noise=instance_dependent_noise,
        max_noise_ratio=max_noise_ratio,
        epsilon_p=epsilon_p,
        epsilon_n=epsilon_n,
    )

    print("class prior", class_prior)
    for i in range(data_num):
        x_1 = X[2 * i]
        x_2 = X[2 * i + 1]
        y_1 = Y[2 * i]
        y_2 = Y[2 * i + 1]

        if digits_included is not None:
            if y_1 not in digits_included or y_2 not in digits_included:
                continue

        if not allow_par and y_1 == y_2:
            continue

        if np.abs(y_1 - y_2) > 2:
            continue

        # Pair the data and compute the BT model probability
        g = (y_1 - y_2) * tempereture
        p = 1 / (1 + np.exp(-g))
        label = np.random.binomial(1, p)
        if label == 0:
            label = 1 - label
            x_1, x_2 = x_2, x_1
            y_1, y_2 = y_2, y_1

        train_X_pair.append(np.concatenate([x_1[None, :], x_2[None, :]]))
        true_label.append(label)
        train_diff.append(y_1 - y_2)
        true_rewards.append(np.array([y_1, y_2]))

    # flip to match the class prior
    flip_prob = 1 - class_prior
    for i in range(len(true_label)):
        if np.random.rand() < flip_prob:
            true_label[i] = 1 - true_label[i]
            train_X_pair[i] = np.concatenate(
                [train_X_pair[i][1][None, :], train_X_pair[i][0][None, :]]
            )
            train_diff[i] = -train_diff[i]
            true_rewards[i] = true_rewards[i][::-1]
        noised_label = noise_fn(true_label[i])
        train_label.append(noised_label)

    return (
        np.array(train_X_pair),
        np.array(train_label),
        np.array(true_label),
        np.array(train_diff),
        np.array(true_rewards),
    )


def flip_all_data(
    X: np.ndarray,
    noised_Y: np.ndarray,
    true_Y: np.ndarray,
    Y_diff: np.ndarray,
    true_rewards: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flip all the x_1 and x_2, resulting in the flip of the label.
    """
    import copy

    print("X shape", X.shape)
    print("noised_Y shape", noised_Y.shape)
    print("true_Y shape", true_Y.shape)
    print("Y_diff shape", Y_diff.shape)
    print("true_rewards shape", true_rewards.shape)
    X_flipped = np.zeros_like(X)
    noised_Y_flipped = np.zeros_like(noised_Y)
    true_Y_flipped = np.zeros_like(true_Y)
    Y_diff_flipped = np.zeros_like(Y_diff)
    true_rewards_flipped = np.zeros_like(true_rewards)
    for i in range(len(X)):
        X_flipped[i][0] = copy.deepcopy(X[i][1])
        X_flipped[i][1] = copy.deepcopy(X[i][0])
        noised_Y_flipped[i] = 1 - noised_Y[i]
        true_Y_flipped[i] = 1 - true_Y[i]
        Y_diff_flipped[i] = -Y_diff[i]
        true_rewards_flipped[i] = true_rewards[i][::-1]
    return (
        X_flipped,
        noised_Y_flipped,
        true_Y_flipped,
        Y_diff_flipped,
        true_rewards_flipped,
    )


def make_dataloader(
    train_data_size,
    test_data_size,
    batch_size,
    tempereture=1,
    digits_included=None,
    allow_par=True,
    add_noise=True,
    symmetric=True,
    instance_dependent_noise=False,
    max_noise_ratio=0.3,
    epsilon_p=0.3,
    epsilon_n=0.3,
    class_prior=0.5,
    flipping=True,
    flip_type="augment",
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    X = dataset.data.numpy()
    Y = dataset.targets.numpy()
    # shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]
    assert len(X) >= train_data_size + test_data_size

    # make noised dataset
    X, noised_Y, true_Y, Y_diff, true_rewards = make_paired_pref_data(
        X,
        Y,
        tempereture=tempereture,
        digits_included=digits_included,
        allow_par=allow_par,
        add_noise=add_noise,
        symmetric=symmetric,
        instance_dependent_noise=instance_dependent_noise,
        max_noise_ratio=max_noise_ratio,
        epsilon_p=epsilon_p,
        epsilon_n=epsilon_n,
        class_prior=class_prior,
    )
    # shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    noised_Y = noised_Y[indices]
    true_Y = true_Y[indices]
    Y_diff = Y_diff[indices]
    true_rewards = true_rewards[indices]
    # split the data
    train_X = X[:train_data_size]
    train_noised_Y = noised_Y[:train_data_size]
    train_true_Y = true_Y[:train_data_size]
    train_Y_diff = Y_diff[:train_data_size]
    train_true_rewards = true_rewards[:train_data_size]
    # print label statistics
    print("noised_Y positive ratio", np.sum(noised_Y) / len(noised_Y))
    print("true_Y positive ratio", np.sum(true_Y) / len(true_Y))
    print("noised_Y matches true_Y", np.sum(noised_Y == true_Y) / len(noised_Y))
    if flipping:
        if flip_type == "augment":
            (
                X_flipped,
                noised_Y_flipped,
                true_Y_flipped,
                Y_diff_flipped,
                true_rewards_flipped,
            ) = flip_all_data(
                train_X, train_noised_Y, train_true_Y, train_Y_diff, train_true_rewards
            )
            train_X = np.concatenate([train_X, X_flipped])
            train_noised_Y = np.concatenate([train_noised_Y, noised_Y_flipped])
            train_true_Y = np.concatenate([train_true_Y, true_Y_flipped])
            train_Y_diff = np.concatenate([train_Y_diff, Y_diff_flipped])
            train_true_rewards = np.concatenate(
                [train_true_rewards, true_rewards_flipped]
            )
            assert len(train_X) == 2 * len(X_flipped)
            print(
                "train_noised_Y positive ratio after flipping",
                np.sum(train_noised_Y) / len(train_noised_Y),
            )
            print(
                "train_true_Y positive ratio after flipping",
                np.sum(train_true_Y) / len(train_true_Y),
            )
            print(
                "train_noised_Y matches train_true_Y after flipping",
                np.sum(train_noised_Y == train_true_Y) / len(train_noised_Y),
            )
        elif flip_type == "random":
            for i in range(len(train_X)):
                if np.random.rand() < 0.5:
                    X_flipped = np.zeros_like(train_X[i])
                    X_flipped[0] = copy.deepcopy(train_X[i][1])
                    X_flipped[1] = copy.deepcopy(train_X[i][0])
                    noised_Y_flipped = not train_noised_Y[i]
                    true_Y_flipped = not train_true_Y[i]
                    Y_diff_flipped = -train_Y_diff[i]
                    true_rewards_flipped = train_true_rewards[i][::-1]
                    train_X[i] = X_flipped
                    train_noised_Y[i] = noised_Y_flipped
                    train_true_Y[i] = true_Y_flipped
                    train_Y_diff[i] = Y_diff_flipped
                    train_true_rewards[i] = true_rewards_flipped
            print(
                "train_noised_Y positive ratio after flipping",
                np.sum(train_noised_Y) / len(train_noised_Y),
            )
            print(
                "train_true_Y positive ratio after flipping",
                np.sum(train_true_Y) / len(train_true_Y),
            )
            print(
                "train_noised_Y matches train_true_Y after flipping",
                np.sum(train_noised_Y == train_true_Y) / len(train_noised_Y),
            )
        else:
            raise ValueError("flip_type must be either 'augment' or 'random'")
    assert train_data_size + test_data_size <= len(X)
    test_X = X[train_data_size : train_data_size + test_data_size]
    test_noised_Y = noised_Y[train_data_size : train_data_size + test_data_size]
    test_true_Y = true_Y[train_data_size : train_data_size + test_data_size]
    test_Y_diff = Y_diff[train_data_size : train_data_size + test_data_size]
    test_true_rewards = true_rewards[train_data_size : train_data_size + test_data_size]
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_X).float(),
        torch.from_numpy(train_noised_Y).float(),
        torch.from_numpy(train_true_Y).float(),
        torch.from_numpy(train_Y_diff).float(),
        torch.from_numpy(train_true_rewards).float(),
    )
    print("train_dataset", len(train_dataset))
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_X).float(),
        torch.from_numpy(test_noised_Y).float(),
        torch.from_numpy(test_true_Y).float(),
        torch.from_numpy(test_Y_diff).float(),
        torch.from_numpy(test_true_rewards).float(),
    )
    print("test_dataset", len(test_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_dataloader, test_dataloader
