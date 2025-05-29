import random
from typing import Literal, Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from data import make_dataloader
from jax import numpy as jnp
from loss import final_test, test, train
from model import CNN
from omegaconf import OmegaConf
from pydantic import BaseModel


class Config(BaseModel):
    project: str = "_mnistpref_test"
    # Training
    algo: Literal["pn", "sympo", "nll"] = "pn"
    loss_type: Literal[
        "sigmoid",
        "logistic",
        "hinge",
        "ramp",
        "symmetric_ramp",
        "square",
        "exponential",
        "unhinged",
    ] = "sigmoid"
    train_data_size: int = 10000
    test_data_size: int = 1000
    batch_size: int = 512
    lr: float = 0.001
    num_epochs: int = 100
    seed: int = 42
    # Model
    clip_min: float = -20
    clip_max: float = 20
    regularize: bool = True
    dropout_ratio: float = 0.2
    save_model: bool = False
    # Data
    digits_included: Optional[list] = None
    allow_par: bool = False
    gen_temperture: float = 1.0
    sigmoid_temperature: float = 1.0
    add_noise: bool = True
    max_noise_ratio: float = 0.3
    instance_dependent_noise: bool = False
    epsilon_p: float = 0.3
    epsilon_n: float = 0.3
    symmetric: bool = True
    class_prior: float = 0.7
    flipping: bool = False
    flip_type: Literal["augment", "random"] = "random"
    # for symmetric_ramp
    slope: float = 0.5


config_dict = OmegaConf.from_cli()
config = Config(**config_dict)

print(config)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(config.seed)
train_loarder, test_loader = make_dataloader(
    config.train_data_size,
    config.test_data_size,
    config.batch_size,
    digits_included=config.digits_included,
    tempereture=config.gen_temperture,
    allow_par=config.allow_par,
    add_noise=config.add_noise,
    symmetric=config.symmetric,
    instance_dependent_noise=config.instance_dependent_noise,
    max_noise_ratio=config.max_noise_ratio,
    epsilon_p=config.epsilon_p,
    epsilon_n=config.epsilon_n,
    flipping=config.algo == "sympo" and config.flipping,
    flip_type=config.flip_type,
    class_prior=config.class_prior,
)
if config.regularize:
    model = CNN(
        config.clip_min,
        config.clip_max,
        use_dropout=True,
        dropout_ratio=config.dropout_ratio,
        use_batchnorm=True,
    )
else:
    model = CNN(
        config.clip_min,
        config.clip_max,
        use_dropout=False,
        use_batchnorm=False,
    )
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

wandb.init(project=config.project, config=config.dict())
model = train(
    model, optimizer, train_loarder, test_loader, torch.device("cuda"), config
)
(
    noisy_label_acc,
    noisy_label_acc_for_correct,
    noisy_label_acc_for_incorrect,
    true_label_acc,
    final_reward_acc,
    reward_diffs,
    rewards,
) = final_test(model, test_loader, torch.device("cuda"), config)
wandb.log(
    {
        "final_true_label_acc": true_label_acc,
        "final_noisy_label_acc": noisy_label_acc,
        "final_noisy_label_acc_for_correct": noisy_label_acc_for_correct,
        "final_noisy_label_acc_for_incorrect": noisy_label_acc_for_incorrect,
        "final_reward_acc": final_reward_acc,
    }
)

reward_diff_data = []
for idx, diff in enumerate(range(-9, 10)):
    if len(reward_diffs[idx]) == 0:
        continue
    reward_diff_data.append(
        [diff, np.mean(reward_diffs[idx]), np.std(reward_diffs[idx])]
    )
wandb.log(
    {
        "reward_diff": wandb.Table(
            data=reward_diff_data,
            columns=["true r_1 - r_2", "pred r_1 - r_2", "std"],
        )
    }
)

reward_data = []
for idx, reward in enumerate(range(0, 10)):
    if len(rewards[idx]) == 0:
        continue
    reward_data.append([reward, np.mean(rewards[idx]), np.std(rewards[idx])])
wandb.log(
    {
        "reward": wandb.Table(
            data=reward_data, columns=["true reward", "pred reward", "std"]
        )
    }
)
if config.save_model:
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")
