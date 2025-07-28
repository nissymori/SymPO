# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Extendable Trainer classes for aligning LLMs.
The specific class that should be used should be specified in the loss file under config/loss.

- The BasicTrainer contains the core methods (e.g., sharding, basic training loop, etc.).
- The SFTTrainer, PairedPreferenceTrainer, and UnpairedPreferenceTrainer all subclass BasicTrainer
  and override the get_batch_metrics() and (optionally) forward() methods.
- The PPOTrainer is a little different, since it also uses a reward model to judge examples before
  updating the policy---note that there is no active sampling, as in standard online PPO.
- The BradleyTerryTrainer is used for training a Bradley-Terry reward model over paired preferences.

The trainer for each loss should subclass either PairedPreferenceTrainer or UnpairedPreferenceTrainer.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = True
import functools
import gc
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoTokenizer

from . import dataloader
# Import loss functions
from .loss import (exp_loss, hinge_loss, logistic_loss, ramp_loss,
                   sigmoid_loss, square_loss, symmetric_ramp_loss,
                   unhinged_loss)
from .models import (AutoModelForBradleyTerry, AutoModelForCausalLM,
                     AutoModelForCausalLMWithValueHead)
from .utils import (delete_dicts, entropy_from_logits, formatted_dict,
                    get_base_model_state_dict_from_peft, masked_mean,
                    masked_var, pad_to_length, rowwise_product)


class BasicTrainer(object):
    policy_hf_model_class = AutoModelForCausalLM
    reference_hf_model_class = AutoModelForCausalLM
    use_reference_model = True

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: DictConfig,
        train_iterator: dataloader.DataLoader,
        eval_iterator: dataloader.DataLoader,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        policy: nn.Module,
        reference_model: Optional[nn.Module] = None,
        num_skip_batches=0,
        **kwargs,
    ):
        """A trainer for a language model, supporting either SFT, HALO, or offline PPO training."""
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.accelerator = accelerator

        self.config = config
        self.run_dir = config.local_run_dir

        self.tokenizer = tokenizer
        self.example_counter = 0
        self.batch_counter = 0

        self.policy = policy
        self.policy_dtype = getattr(torch, config.model.policy_dtype)

        self.reference_model = reference_model
        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_skip_batches = num_skip_batches  # when loading from checkpoint

        self.reward_model = kwargs.get("reward_model", None)
        self.reward_tokenizer = kwargs.get("reward_tokenizer", None)
        if self.reward_model is not None:
            assert (
                self.reward_tokenizer is not None
            ), "reward_tokenizer must be provided when using reward_model"
            self.reward_model.eval()

        self.prepare_accelerator()

    def prepare_accelerator(self):
        """Prepare the Accelerator."""
        self.policy, self.reference_model, self.optimizer, self.scheduler = (
            self.accelerator.prepare(
                self.policy, self.reference_model, self.optimizer, self.scheduler
            )
        )

        if self.reward_model:
            self.reward_model = self.accelerator.prepare(self.reward_model)

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        """Compute the token-level log probabilities of the given labels under the given logits."""
        # ignoring vocab size, batch size x length should be equal
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        distribution_logps = logits.float().log_softmax(-1)
        per_token_logps = torch.gather(
            distribution_logps, dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        return per_token_logps * loss_mask

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy."""
        with self.accelerator.autocast():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.model.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=self.config.top_p,
            )

        policy_output = pad_to_length(
            policy_output, self.config.model.max_length, self.tokenizer.pad_token_id
        )
        policy_output = self.accelerator.gather(policy_output)
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        return policy_output_decoded

    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            batch: batch of data, mapping keys to Tensors
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (microbatch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (microbatch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (microbatch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (microbatch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the losses, one for each example (sif chosen_only or rejected_only, only n/2 losses).
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively, for reporting.
            Note that rejected responses do not factor into the loss, only the reward calculation.
        """
        raise NotImplementedError

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = None
    ) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.

        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'

        Returns:
            A tuple of a scalar loss and a dict of metrics.
        """
        raise NotImplementedError

    def eval(self) -> Dict[str, Dict]:
        """
        Run evaluation on all the examples in the test data and return the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset.
        It does not compare two models to eacch other.

        Returns:
            A dict of form:
            {
                'metadata': the Hydra config
                'results': a dict of batch metrics (averaged across all of the test data)
            }
        """
        self.accelerator.print(
            f"Running evaluation after {self.example_counter} train examples"
        )
        self.policy.eval()

        if self.reference_model is not None:
            self.reference_model.eval()

        all_eval_metrics = defaultdict(list)

        for eval_batch in (
            tqdm(self.eval_iterator, desc="Computing eval metrics")
            if self.accelerator.is_main_process
            else self.eval_iterator
        ):
            eval_batch = {
                k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in eval_batch.items()
            }
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(eval_batch, mode="eval")

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(
                    torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist()
                )

        # Compute mean metrics
        mean_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        delete_dicts(eval_batch, eval_metrics, all_eval_metrics)
        self.free_memory()

        if self.accelerator.is_main_process and self.config.wandb.enabled:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        else:
            results = None

        results = {
            "metadata": OmegaConf.to_container(self.config),
            "results": formatted_dict(mean_eval_metrics),
        }

        return results

    def train(self):
        """Begin either SFT or HALO training, with periodic evaluation. This is subclassed when implementing PPO."""

        self.accelerator.print(
            f"Using {self.config.optimizer} optimizer with learning rate {self.config.lr}"
        )

        if self.reference_model is not None:
            self.reference_model.eval()

        last_log = None
        batch_metrics = defaultdict(list)

        for batch in self.train_iterator:
            if self.batch_counter < self.num_skip_batches:
                self.batch_counter += 1
                self.example_counter += self.config.model.batch_size
                continue

            # EVALUATION
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                results = self.eval()

                if self.example_counter > 0:
                    if self.config.debug:
                        self.accelerator.print("skipping save in debug mode")
                    elif self.config.intermediate_checkpoints:
                        output_dir = os.path.join(
                            self.run_dir, f"step-{self.example_counter}"
                        )
                        self.accelerator.print(
                            f"creating checkpoint to write to {output_dir}..."
                        )
                        self.save(output_dir, results["results"], final_save=False)

                self.accelerator.print(results["results"])
                delete_dicts(results)

            # TRAINING
            self.policy.train()
            accumulated = 0
            start_time = time.time()

            with self.accelerator.accumulate(self.policy):
                batch = {
                    k: (
                        v.to(self.accelerator.device)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }
                loss, metrics = self.get_batch_metrics(batch)
                self.accelerator.backward(loss)

                for k, v in metrics.items():
                    batch_metrics[k].extend(
                        torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist()
                    )

                grad_norm = self.accelerator.clip_grad_norm_(
                    self.policy.parameters(), self.config.model.max_grad_norm
                )
                batch_metrics["grad_norm"].extend(
                    torch.as_tensor(grad_norm)
                    .reshape(-1)
                    .float()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accumulated += 1

            step_time = time.time() - start_time
            examples_per_second = self.config.model.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)

            self.batch_counter += 1
            self.example_counter += self.config.model.batch_size

            if (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = sum(v) / len(v)

                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                self.accelerator.print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.accelerator.is_main_process:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
                batch_metrics = defaultdict(list)
            else:
                self.accelerator.print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )

            delete_dicts(batch, metrics, batch_metrics, mean_train_metrics)

            if accumulated >= self.config.model.gradient_accumulation_steps:
                self.free_memory()
                accumulated = 0

    def save(
        self,
        output_dir: Optional[str] = None,
        metrics: Optional[Dict] = {},
        final_save=True,
    ):
        """Save tokenizer, policy model, optimizer, scheduler state to disk."""
        self.accelerator.print(f"Saving...")
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, f"step-{self.example_counter}")

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            self.accelerator.print(f"Saving tokenizer...")
            self.tokenizer.save_pretrained(output_dir)

            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                metrics["counter"] = self.example_counter
                json.dump(metrics, f)

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving state...")
        optimizer = self.accelerator.unwrap_model(self.optimizer)
        scheduler = self.accelerator.unwrap_model(self.scheduler)
        if self.accelerator.is_main_process:
            optimizer_state = {
                "state_dict": optimizer.state_dict(),
                "class": optimizer.__class__.__name__,
            }
            torch.save(optimizer_state, os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving model...")

        if self.config.model.use_peft and final_save:
            state_dict = get_base_model_state_dict_from_peft(
                self.accelerator.get_state_dict(self.policy),
                self.config.model.peft.lora_alpha,
                self.config.model.peft.lora_r,
            )
            unwrapped_model = self.accelerator.unwrap_model(self.policy).base_model
        else:
            state_dict = self.accelerator.get_state_dict(self.policy)
            unwrapped_model = self.accelerator.unwrap_model(self.policy)

        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )

        self.accelerator.wait_for_everyone()

    def free_memory(self):
        torch.cuda.empty_cache()
        self.accelerator.free_memory()
        gc.collect()


class SFTTrainer(BasicTrainer):
    use_reference_model = False

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = "train"
    ):
        """Compute the loss and other metrics for the given batch of inputs.

        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids',
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'eval', 'sample'
        """
        metrics = {}

        with self.accelerator.autocast():
            policy_chosen_logits = self.policy(
                batch["target_combined_input_ids"],
                attention_mask=batch["target_combined_attention_mask"],
            ).logits.to(self.policy_dtype)

            policy_chosen_logps = self.get_batch_logps(
                policy_chosen_logits, batch["target_labels"]
            )
            policy_chosen_logps = policy_chosen_logps.view(-1)
            losses = -policy_chosen_logps

        # Gather losses and logps from all processes
        total_nonzero_elements = self.accelerator.gather(
            (policy_chosen_logps != 0).sum().detach()
        ).sum()
        metrics[f"logps_{mode}/chosen"] = (
            self.accelerator.gather(policy_chosen_logps.detach()).sum()
            / total_nonzero_elements
        )
        metrics[f"loss/{mode}"] = (
            self.accelerator.gather(losses.sum().detach()).sum()
            / total_nonzero_elements
        )

        del policy_chosen_logits, policy_chosen_logps

        return losses.sum(), metrics


class UnpairedPreferenceTrainer(BasicTrainer):
    use_reference_model = True

    """A trainer for any loss that doesn't use paired preference, like KTO."""

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        use_cache: bool = False,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor
    ]:
        """Run the given model on the given batch of inputs.

        Returns:
            chosen_logps: log probabilities of chosen examples
            rejected_logps: log probabilities of rejected examples
            use_cache: if true, expecte to get cached logprobs from the model
        """
        with self.accelerator.autocast():
            if use_cache:
                all_logps = (
                    model(batch["target_combined_input_ids"])
                    .to(self.policy_dtype)
                    .to(self.accelerator.device)
                )
            else:
                all_logits = model(
                    batch["target_combined_input_ids"],
                    attention_mask=batch["target_combined_attention_mask"],
                ).logits.to(self.policy_dtype)

                all_logps = self.get_batch_logps(all_logits, batch["target_labels"])

        assert all_logps.shape[0] == len(batch["status"])
        chosen_idx = [
            i for i in range(all_logps.shape[0]) if batch["status"][i] == "chosen"
        ]
        rejected_idx = [
            i for i in range(all_logps.shape[0]) if batch["status"][i] == "rejected"
        ]

        chosen_logps = all_logps[chosen_idx, ...]
        rejected_logps = all_logps[rejected_idx, ...]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = "train"
    ):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(
                self.policy, batch
            )
            losses, chosen_rewards, rejected_rewards = self.loss(
                batch, policy_chosen_logps, policy_rejected_logps
            )
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(
                    self.reference_model,
                    batch,
                    use_cache=self.config.cache_reference_logprobs,
                )
            losses, chosen_rewards, rejected_rewards = self.loss(
                batch,
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

        # all_gather treats empty lists/tensors poorly, and empty lists can occur because a batch can contain all chosen or all rejected example
        # therefore, concatenate chosen + rejected rewards before all_gather
        combined_rewards = torch.cat(
            (chosen_rewards.detach(), rejected_rewards.detach()), 0
        )
        combined_statuses = torch.Tensor(
            [1] * len(chosen_rewards) + [0] * len(rejected_rewards)
        ).to(self.accelerator.device)

        all_rewards = self.accelerator.gather(combined_rewards.detach())
        all_statuses = self.accelerator.gather(combined_statuses.detach())
        chosen_rewards_idx = [
            i for i in range(len(all_statuses)) if all_statuses[i].item() == 1
        ]
        rejected_rewards_idx = [
            i for i in range(len(all_statuses)) if all_statuses[i].item() == 0
        ]

        metrics[f"rewards_{mode}/chosen"] = all_rewards[chosen_rewards_idx]
        metrics[f"rewards_{mode}/rejected"] = all_rewards[rejected_rewards_idx]
        metrics[f"rewards_{mode}/margins"] = torch.Tensor(
            [
                (
                    all_rewards[chosen_rewards_idx].mean().nan_to_num(0)
                    - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)
                ).item()
            ]
        )
        metrics[f"loss/{mode}"] = self.accelerator.gather(losses.mean().detach()).mean()

        del policy_chosen_logps, policy_rejected_logps
        del (
            combined_rewards,
            combined_statuses,
            all_rewards,
            all_statuses,
            chosen_rewards_idx,
            rejected_rewards_idx,
            all_devices_losses,
        )

        if self.reference_model:
            del reference_chosen_logps, reference_rejected_logps

        return losses.sum(), metrics


class PairedPreferenceTrainer(BasicTrainer):
    use_reference_model = True

    """A trainer for any loss that uses paired preference, like DPO."""

    def concatenated_inputs(
        self, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor. The first half is chosen outputs, the second half is rejected.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (microbatch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(
            batch["chosen_combined_input_ids"].shape[1],
            batch["rejected_combined_input_ids"].shape[1],
        )
        concatenated_batch = {}

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if "_labels" in k else 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if "_labels" in k else 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )

        return concatenated_batch

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        use_cache: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        Return two tensors of shape (batch size), one of the chosen examples, another of the rejected ones.

        Returns:
         chosen_logps: log probabilities of chosen examples
         rejected_logps: log probabilities of rejected examples
         use_cache: if true, expecte to get cached logprobs from the model
        """
        with self.accelerator.autocast():
            concatenated_batch = self.concatenated_inputs(batch)

            if use_cache:
                all_logps = (
                    model(batch["concatenated_combined_input_ids"])
                    .to(self.policy_dtype)
                    .to(self.accelerator.device)
                )
            else:
                all_logits = model(
                    concatenated_batch["concatenated_combined_input_ids"],
                    attention_mask=concatenated_batch[
                        "concatenated_combined_attention_mask"
                    ],
                ).logits.to(self.policy_dtype)

                all_logps = self.get_batch_logps(
                    all_logits, concatenated_batch["concatenated_labels"]
                )

        chosen_logps = all_logps[: batch["chosen_combined_input_ids"].shape[0], ...]
        rejected_logps = all_logps[batch["chosen_combined_input_ids"].shape[0] :, ...]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = "train"
    ):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(
                self.policy, batch
            )
            losses, chosen_rewards, rejected_rewards = self.loss(
                batch, policy_chosen_logps, policy_rejected_logps
            )
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(
                    self.reference_model,
                    batch,
                    use_cache=self.config.cache_reference_logprobs,
                )
            losses, chosen_rewards, rejected_rewards = self.loss(
                batch,
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

        # accuracy calculated on paired examples (for apples-to-apples comparison with UnpairedPreferenceTrainer)

        # for (i, j) in batch['pairs']: the label is 1 if i > j, 0 if j > i
        noisy_preference_labels = batch["noisy_preference_labels"].to(
            self.accelerator.device
        )[
            :, 0
        ]  # shape (batch_size, 1)
        true_preference_labels = batch["true_preference_labels"].to(
            self.accelerator.device
        )[
            :, 0
        ]  # shape (batch_size, 1)
        assert torch.all(
            torch.logical_or(noisy_preference_labels == 0, noisy_preference_labels == 1)
        )
        assert torch.all(
            torch.logical_or(true_preference_labels == 0, true_preference_labels == 1)
        )
        preference_accuracies = (
            noisy_preference_labels == true_preference_labels
        ).float()

        pred_labels = (chosen_rewards > rejected_rewards).float()
        true_label_accuracies = (pred_labels == true_preference_labels).float()
        noisy_label_accuracies = (pred_labels == noisy_preference_labels).float()

        metrics[f"rewards_{mode}/chosen"] = self.accelerator.gather(
            chosen_rewards.detach()
        )
        metrics[f"rewards_{mode}/rejected"] = self.accelerator.gather(
            rejected_rewards.detach()
        )
        metrics[f"rewards_{mode}/true_label_accuracies"] = self.accelerator.gather(
            true_label_accuracies.detach()
        )
        metrics[f"rewards_{mode}/noisy_label_accuracies"] = self.accelerator.gather(
            noisy_label_accuracies.detach()
        )
        metrics[f"rewards_{mode}/margins"] = self.accelerator.gather(
            (chosen_rewards - rejected_rewards).detach()
        )
        metrics[f"logps_{mode}/rejected"] = self.accelerator.gather(
            policy_rejected_logps.detach()
        )
        metrics[f"logps_{mode}/chosen"] = self.accelerator.gather(
            policy_chosen_logps.detach()
        )
        metrics[f"loss/{mode}"] = self.accelerator.gather(losses.mean().detach()).mean()

        del (
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logps,
            policy_rejected_logps,
            true_label_accuracies,
            noisy_label_accuracies,
            pred_labels,
        )
        if self.reference_model:
            del reference_chosen_logps, reference_rejected_logps

        return losses.sum(), metrics


class DPOTrainer(PairedPreferenceTrainer):
    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.config.loss.beta * (
            policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1)
        )
        rejected_rewards = self.config.loss.beta * (
            policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1)
        )

        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class SymPOTrainer(PairedPreferenceTrainer):

    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        chosen_rewards = self.config.loss.beta * (
            policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1)
        )
        rejected_rewards = self.config.loss.beta * (
            policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1)
        )

        if self.config.loss.clip:
            rejected_rewards = torch.clamp(
                rejected_rewards,
                min=-self.config.loss.clip_value,
                max=self.config.loss.clip_value,
            )  # [-clip, clip]
            chosen_rewards = torch.clamp(
                chosen_rewards,
                min=-self.config.loss.clip_value,
                max=self.config.loss.clip_value,
            )  # [-clip, clip]

        logits = chosen_rewards - rejected_rewards

        label = torch.ones(logits.shape[0]).to(self.accelerator.device)

        if self.config.loss.loss_type == "sigmoid":
            losses = sigmoid_loss(label, logits)
        elif self.config.loss.loss_type == "ramp":
            losses = ramp_loss(label, logits)
        elif self.config.loss.loss_type == "symmetric_ramp":
            losses = symmetric_ramp_loss(label, logits)
        elif self.config.loss.loss_type == "unhinged":
            losses = unhinged_loss(label, logits)
        elif self.config.loss.loss_type == "square":
            losses = square_loss(label, logits)
        elif self.config.loss.loss_type == "hinge":
            losses = hinge_loss(label, logits)
        else:
            raise ValueError(f"Invalid loss type: {self.config.loss.loss_type}")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class ROPOTrainer(PairedPreferenceTrainer):

    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_rewards = self.config.loss.beta * (
            policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1)
        )
        rejected_rewards = self.config.loss.beta * (
            policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1)
        )
        logits = chosen_rewards - rejected_rewards
        label = torch.ones(logits.shape[0]).to(self.accelerator.device)

        dpo_weight = 4 * self.config.loss.alpha / (self.config.loss.alpha + 1) ** 2
        reg_weight = 4 * self.config.loss.alpha**2 / (self.config.loss.alpha + 1) ** 2

        dpo_loss = -F.logsigmoid(logits)
        reg_loss = sigmoid_loss(label, logits)

        losses = dpo_weight * dpo_loss + reg_weight * reg_loss

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class RDPOTrainer(PairedPreferenceTrainer):

    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_rewards = self.config.loss.beta * (
            policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1)
        )
        rejected_rewards = self.config.loss.beta * (
            policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1)
        )
        logits = chosen_rewards - rejected_rewards

        rdpo_loss = -(1 - self.config.noise_ratio) * F.logsigmoid(logits) + self.config.noise_ratio * F.logsigmoid(-logits)
        rdpo_loss = rdpo_loss / (1 - 2 * self.config.noise_ratio)
        return rdpo_loss, chosen_rewards.detach(), rejected_rewards.detach()


class CDPOTrainer(PairedPreferenceTrainer):
    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CDPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.config.loss.beta * (
            policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1)
        )
        rejected_rewards = self.config.loss.beta * (
            policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1)
        )

        forward_losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        reverse_losses = -F.logsigmoid(rejected_rewards - chosen_rewards)
        losses = (
            1 - self.config.noise_ratio
        ) * forward_losses + self.config.noise_ratio * reverse_losses

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class IPOTrainer(PairedPreferenceTrainer):
    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the IPO loss for a batch of policy and reference model token-level log probabilities."""
        prompt_lengths = (
            (batch["prompt_input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .clamp(min=1)
        )

        chosen_combined_lengths = (
            (batch["chosen_combined_input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .clamp(min=1)
        )
        chosen_lengths = (chosen_combined_lengths - prompt_lengths).clamp(min=1)

        rejected_combined_lengths = (
            (batch["rejected_combined_input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .clamp(min=1)
        )
        rejected_lengths = (rejected_combined_lengths - prompt_lengths).clamp(min=1)

        # must average over the probabilities
        policy_chosen_logps = policy_chosen_logps.sum(-1) / chosen_lengths
        policy_rejected_logps = policy_rejected_logps.sum(-1) / rejected_lengths
        reference_chosen_logps = reference_chosen_logps.sum(-1) / chosen_lengths
        reference_rejected_logps = reference_rejected_logps.sum(-1) / rejected_lengths

        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps

        losses = (
            chosen_rewards - rejected_rewards - (1 / (2 * self.config.loss.tau))
        ).pow(2)

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class SimPOTrainer(PairedPreferenceTrainer):
    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy and reference model token-level log probabilities."""
        prompt_lengths = (
            (batch["prompt_input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .clamp(min=1)
        )

        chosen_combined_lengths = (
            (batch["chosen_combined_input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .clamp(min=1)
        )
        chosen_lengths = (chosen_combined_lengths - prompt_lengths).clamp(min=1)
        chosen_rewards = policy_chosen_logps.sum(-1) / chosen_lengths

        rejected_combined_lengths = (
            (batch["rejected_combined_input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .clamp(min=1)
        )
        rejected_lengths = (rejected_combined_lengths - prompt_lengths).clamp(min=1)
        rejected_rewards = policy_rejected_logps.sum(-1) / rejected_lengths

        losses = -F.logsigmoid(
            self.config.loss.beta
            * (chosen_rewards - rejected_rewards - self.config.loss.gamma_beta_ratio)
        )

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class SLiCTrainer(PairedPreferenceTrainer):
    use_reference_model = False

    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SLIC loss as defined by Zhao et al. in https://arxiv.org/pdf/2305.10425.pdf

        Calibration loss defined as:
            L(x, y) := max(0, beta - log p_policy(y_chosen|x) + log p_rejected(y|x))
        For the cross-entropy loss, just use the NLL of the chosen sequence (equivalent to SFT).
        """
        cal_loss = torch.clamp(
            self.config.loss.beta
            - policy_chosen_logps.sum(-1)
            + policy_rejected_logps.sum(-1),
            min=0,
        )
        reg_loss = -policy_chosen_logps

        losses = cal_loss + self.config.loss.lambda_coef * reg_loss

        chosen_rewards = policy_chosen_logps.detach()
        rejected_rewards = policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards
