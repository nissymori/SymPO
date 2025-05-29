# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains the functions for loading data.
Each function of the form get_{dataset_name} (e.g., get_shp, get_oasst, etc.) will return a dict of Example objects, indexed by the prompt for the text.

Each Example object will contain
- the prompt
- a list L of generations
- the index in L of the generation that should be the finetuning target
- a list S of the scores for the generations
- for preference feedback data: pairs of indices (i,j) in L, where generation i is preferable to generation j
- for binary feedback data: whether each generation is desirable/chosen or undesirable/rejected
- whether to truncate the beginning or end if the maximum number of tokens is exceeded
- the dataset name
- the unformatted prompt
"""

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence

from .utils import delete_dict, on_rank0, rank0_print


@dataclass
class Example:
    """
    Class for an example in a preference or SFT dataset. If you want each prompt to be uniquely associated with an Example instance, save it in a dict.
    """

    prompt: List = field(
        default_factory=list
    )  # list of turns, each with two keys: "role" and "content"
    generations: List = field(
        default_factory=list
    )  # list of list of turns (the output sequences to predict)
    sft_index: int = (
        -1
    )  # which response in self.generations should be generated for SFT
    scores: List[float] = field(default_factory=list)  # score for each generation
    pairs: List[Tuple[int, int]] = field(
        default_factory=list
    )  # for preference feedback data: indices in responses, where i > j in pair (i,j) is a preference
    desirable: List[bool] = field(
        default_factory=list
    )  # for binary feedback data: whether the generation at the corresponding index in self.generations is desirable
    truncation_mode: str = (
        "keep_end"  # if truncation needed, keep the beginning (keep_start) or end (keep_end) (only override default for SHP)
    )
    dataset_name: str = ""
    original_prompt: str = (
        ""  # the unformatted prompt (needed to recover instruction for AlpacaEval)
    )
    noisy_preference_labels: List[int] = field(
        default_factory=list
    )  # since pair (i,j) is a preference, the noisy label is always 1, which means i > j
    true_preference_labels: List[int] = field(
        default_factory=list
    )  # the true preference labels

    def num_generations(self):
        return len(self.generations)

    def remove_extra_spaces(self):
        """
        Remove double spaces in the prompt and generations to standardize spacing.
        """

        def clean(text: str) -> str:
            return re.sub(r"[ \t]{2,}", " ", text)

        # Clean the prompt
        for turn in self.prompt:
            turn["content"] = clean(turn["content"])

        # Clean the generations
        for x in self.generations:
            for turn in x:
                turn["content"] = clean(turn["content"])


class Dataset:
    """
    A collection of Example instances, indexed by prompt.
    """

    def __init__(self, name):
        self.name = name
        self.data = defaultdict(Example)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("key must be a string")

        if not isinstance(value, Example):
            raise ValueError("value must be a Example")

        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def get_alpacaeval(split: str) -> Dataset:
    """
    Load the AlpacaEval dataset (for evaluation only) and convert it into a Dataset.

    Args:
        - split: must be 'test'; otherwise error will be thrown

    Returns:
        A Dataset instance.
    """
    if split == "test":
        split = "eval"
    else:
        raise ValueError("alpacaeval is only for evaluation")

    rank0_print(f"Loading AlpacaEval dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc="Processing AlpacaEval")

    data = Dataset("alpacaeval")

    for row in dataset:
        conversation = [{"role": "user", "content": row["instruction"]}]
        data[row["instruction"]].prompt = conversation
        data[row["instruction"]].generations.append(
            [{"role": "assistant", "content": row["output"]}]
        )
        data[row["instruction"]].dataset_name = row["dataset"]
        data[row["instruction"]].original_prompt = row["instruction"]
        data[row["instruction"]].sft_index = 0

    return data


def get_sampled_data(samples_path: str, split: str) -> Dataset:
    """
    Load samples generated by train.sample and convert it into a Dataset.
    """
    rank0_print(f"Loading samples from {samples_path}...")

    # Load all sample data
    with open(samples_path, "r") as f:
        sample_data = json.load(f)

    data = Dataset("samples")

    for sample in sample_data:
        if sample.get("split", split) != split:
            continue

        prompt_key = str(sample["prompt_id"]) + " " + str(sample["sample_id"])
        data[prompt_key].prompt = sample["prompt"]
        data[prompt_key].generations.append(
            [{"role": "assistant", "content": sample["output"]}]
        )
        data[prompt_key].dataset_name = sample["dataset"]
        data[prompt_key].sft_index = 0

    return data


def assert_noise_ratio(data: Dataset, noise_ratio: float):
    noisy_labels = [
        label for key in data for label in data[key].noisy_preference_labels
    ]
    true_labels = [label for key in data for label in data[key].true_preference_labels]

    matches = sum(n == t for n, t in zip(noisy_labels, true_labels))
    data_noise_ratio = 1 - matches / len(noisy_labels)

    assert (
        data_noise_ratio <= noise_ratio + 0.01
    ), f"Noise ratio is not consistent true noise ratio: {noise_ratio} data noise ratio: {data_noise_ratio}"
    assert (
        data_noise_ratio >= noise_ratio - 0.01
    ), f"Noise ratio is not consistent true noise ratio: {noise_ratio} data noise ratio: {data_noise_ratio}"
    print("len(noisy_labels)", len(noisy_labels))
    print("data_noise_ratio", data_noise_ratio)
    print("noise_ratio", noise_ratio)


def flip_data(example: Example, noise_ratio: float):
    flip_label = random.random() < noise_ratio
    example_length = len(example.pairs)
    example.noisy_preference_labels.append(
        1
    )  # the noisy label is always 1, which means i > j
    i, j = example.pairs[-1]  # here i>j
    if flip_label:
        example.pairs[-1] = (j, i)  # flip the preference pair
        example.true_preference_labels.append(
            0
        )  # the true label is 0, which means j > i
        assert example.pairs[-1][0] == j and example.pairs[-1][1] == i
    else:
        example.pairs[-1] = (i, j)  # keep the preference pair
        example.true_preference_labels.append(
            1
        )  # the true label is 1, which means i > j
    assert len(example.pairs) == example_length
    assert len(example.true_preference_labels) == example_length
    assert len(example.noisy_preference_labels) == example_length
    return example


def get_hh(
    split: str, only_helpful=False, only_harmless=False, noise_ratio: float = 0.0
) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - only_helpful: only the helpfulness data
        - only_harmless: only the harmlessness data
        - noise_ratio: the ratio of noisy labels to true labels

    Returns:
        A Dataset instance.
    """
    if only_helpful:
        dataset = datasets.load_dataset(
            "Anthropic/hh-rlhf", split=split, data_dir="helpful-base"
        )
        data = Dataset("Anthropic-HH-helpful")
    elif only_harmless:
        dataset = datasets.load_dataset(
            "Anthropic/hh-rlhf", split=split, data_dir="harmless-base"
        )
        data = Dataset("Anthropic-HH-harmless")
    else:
        rank0_print(f"Loading HH dataset ({split} split) from Huggingface...")
        dataset = datasets.load_dataset("Anthropic/hh-rlhf", split=split)
        data = Dataset("Anthropic-HH")

    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc="Processing HH")

    def split_prompt_and_responses(ex):
        parts = re.split(r"\n\nHuman: |\n\nAssistant: ", ex["chosen"])
        conversation = []
        for i, part in enumerate(parts[1:]):  # Skip the first empty part
            role = "user" if i % 2 == 0 else "assistant"
            conversation.append({"role": role, "content": part.strip()})
        chosen_response = conversation.pop()["content"]
        rejected_response = ex["rejected"].split("\n\nAssistant: ")[-1].strip()
        return conversation, chosen_response, rejected_response

    for row in dataset:
        conversation, chosen, rejected = split_prompt_and_responses(row)
        prompt_key = " ".join(
            [turn["content"] for turn in conversation]
        )  # Use full conversation as key
        i, j = (
            data[prompt_key].num_generations(),
            data[prompt_key].num_generations() + 1,
        )

        data[prompt_key].prompt = conversation
        data[prompt_key].generations.append([{"role": "assistant", "content": chosen}])
        data[prompt_key].generations.append(
            [{"role": "assistant", "content": rejected}]
        )
        data[prompt_key].pairs.append((i, j))
        data[prompt_key].sft_index = 0

        if only_helpful:
            data[prompt_key].dataset_name = "hh_helpful"
        elif only_harmless:
            data[prompt_key].dataset_name = "hh_harmless"
        else:
            data[prompt_key].dataset_name = "hh"

        data[prompt_key].remove_extra_spaces()

        data[prompt_key] = flip_data(data[prompt_key], noise_ratio)
    assert_noise_ratio(data, noise_ratio)

    return data


def get_hh_helpful(split: str, noise_ratio: float = 0.0) -> Dataset:
    return get_hh(split, only_helpful=True, noise_ratio=noise_ratio)


def get_hh_harmless(split: str, noise_ratio: float = 0.0) -> Dataset:
    return get_hh(split, only_harmless=True, noise_ratio=noise_ratio)


def get_ultrabin(split: str, noise_ratio: float = 0.0) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'

    Returns:
        A Dataset instance.
    """
    if split == "train":
        split = "train_prefs"
    elif split == "test":
        split = "test_prefs"
    else:
        raise ValueError("Split must be either 'train' or 'test'")

    rank0_print(f"Loading Ultra Binarized dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", split=split
    )
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc="Processing Ultrachat Binarized")

    data = Dataset("ultrabin")

    for row in dataset:
        # Convert the prompt into the new format
        conversation = [{"role": "user", "content": row["prompt"]}]

        # Get the chosen and rejected responses
        chosen_response = row["chosen"][-1]["content"]
        rejected_response = row["rejected"][-1]["content"]

        # Create a unique key for this example (using the prompt)
        key = row["prompt"]

        # Update the dataset
        data[key].prompt = conversation
        data[key].generations.append(
            [{"role": "assistant", "content": chosen_response}]
        )
        data[key].generations.append(
            [{"role": "assistant", "content": rejected_response}]
        )
        i, j = data[key].num_generations() - 2, data[key].num_generations() - 1
        data[key].pairs.append((i, j))
        data[key].sft_index = i  # The chosen response is the SFT target
        data[key].dataset_name = data.name
        data[key].truncation_mode = "keep_start"
        data[key].remove_extra_spaces()
        data[key] = flip_data(data[key], noise_ratio)
    assert_noise_ratio(data, noise_ratio)

    return data


def get_ultrafeedback_armorm(split: str, noise_ratio: float = 0.0) -> Dataset:
    rank0_print(
        f"Loading ultrafeedback_armorm dataset ({split} split) from Huggingface..."
    )
    dataset = datasets.load_dataset(
        "princeton-nlp/llama3-ultrafeedback-armorm", split=split
    )
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc="Processing ultrafeedback armorm")

    data = Dataset("ultrafeedback_armorm")

    for row in dataset:
        # Convert the prompt into the new format
        conversation = [{"role": "user", "content": row["prompt"]}]

        # Create a unique key for this example (using the prompt)
        key = row["prompt"]

        # Update the dataset
        data[key].prompt = conversation
        data[key].generations.append([row["chosen"][-1]])
        data[key].generations.append([row["rejected"][-1]])
        i, j = data[key].num_generations() - 2, data[key].num_generations() - 1
        data[key].pairs.append((i, j))
        data[key].sft_index = i  # The chosen response is the SFT target
        data[key].dataset_name = data.name
        data[key].truncation_mode = "keep_start"
        data[key].remove_extra_spaces()
        data[key] = flip_data(data[key], noise_ratio)
    assert_noise_ratio(data, noise_ratio)
    return data


def get_ultrachat(split: str, noise_ratio: float = 0.0) -> Dataset:
    rank0_print(f"Loading ultrachat dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset(
        "HuggingFaceH4/ultrachat_200k", split=f"{split}_sft"
    )
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc="Processing ultrachat")

    data = Dataset("ultrachat")

    for row in dataset:
        key = row["prompt"]
        data[key].prompt = [row["messages"][0]]
        data[key].generations.append(row["messages"][1:])
        data[key].sft_index = 0
        data[key].dataset_name = data.name
        data[key].truncation_mode = "keep_start"
        data[key].remove_extra_spaces()
        data[key] = flip_data(data[key], noise_ratio)
    assert_noise_ratio(data, noise_ratio)
    return data


class DataLoader:
    """
    The base data loader class, similar to the one from the DPO repo.
    Subclass this and overwrite the __iter__ method as needed, since the batcch elements will be different depending
    on whether you're doing SFT, aligning with a pairwise loss like DPO, or alignment with an unpaired loss like KTO.
    """

    def __init__(
        self,
        dataset_names: List[str],
        tokenizer,
        process_index: int = 0,
        num_processes: int = 1,
        split: str = "train",
        microbatch_size: int = 1,
        max_length: int = 512,
        max_prompt_length: int = 128,
        max_prompt_count: int = None,
        n_epochs: Optional[int] = None,
        n_examples: Optional[int] = None,
        seed: int = 0,
        control_tokens: Dict = {},
        **kwargs,
    ):

        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.process_index = process_index
        self.num_processes = num_processes
        self.control_tokens = control_tokens
        self.split = split
        self.microbatch_size = microbatch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count
        self.kwargs = kwargs
        self.noise_ratio = kwargs.get("noise_ratio", 0.0)

        assert (
            n_epochs is not None or n_examples is not None
        ), "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples

        self.full_data = {}  # a dict of Examples

        for name in dataset_names:
            if f"get_{name}" in globals():
                dataset = globals()[f"get_{name}"](split, noise_ratio=self.noise_ratio)
                self.full_data.update(dataset.data)
            else:
                try:
                    with open(name, "r") as f:
                        data = json.load(f)

                        if data[0]["type"] == "sample":
                            dataset = get_sampled_data(name, split)
                        elif data[0]["type"].endswith("feedback"):
                            dataset = get_feedback(name, split)
                        else:
                            raise IOError("unrecognized data type")

                        self.full_data.update(dataset.data)
                except:
                    raise IOError(f"could not load {name}")

        self.num_training_steps = self.get_num_training_steps()

    def collate(self, batch: Dict[str, List]) -> Dict:
        """
        Takes a list of examples and returns a batch of examples with consistent padding across all processes.
        Uses a fixed maximum length for padding to ensure consistency across batches and processes.
        """
        if self.tokenizer.pad_token_id is None:
            raise Exception("tokenizer's pad_token_id is not specified")

        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if "prompt" in k:
                    # flip prompt so that you are padding to the beginning
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Always pad to max_length for consistency across processes
                max_len = self.max_prompt_length if "prompt" in k else self.max_length

                padded_sequences = []
                for seq in to_pad:
                    if len(seq) > max_len:
                        padded_seq = seq[:max_len]
                    else:
                        padding_size = max_len - len(seq)
                        padding = torch.full(
                            (padding_size,), padding_value, dtype=seq.dtype
                        )
                        padded_seq = torch.cat([seq, padding])
                    padded_sequences.append(padded_seq)

                padded_batch[k] = torch.stack(padded_sequences)
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def tokenize_batch_element(
        self,
        conversation: List[Dict[str, str]],
        generation: str,
        truncation_mode: str,
        prefix: str = "target",
    ) -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - conversation: list of previous turns, each resembling dict {"role": "assistant", "content": generation}
        - generation: list of current turns, each resembling dict {"role": "assistant", "content": generation}
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of prompt respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt and the concatenation of the two on all relevant elements (e.g., tokens,
            attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the concatenated
            elements will have keys starting with '{prefix}_combined_'. 'prompt' will map to the raw conversation history,
            as a list of dicts, and the prefix key alone will map to the untemplated output.
        """
        untruncated_prompt_string = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )  # for inference-time generation

        filter_out_bos_eos = lambda x: [
            t
            for t in x
            if t
            not in [
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            ]
        ]
        # truncate the prompt if necessary
        total_length = 0

        # truncate history to fit in self.max_prompt_length
        for i, turn in enumerate(conversation):
            content_token_ids = filter_out_bos_eos(
                self.tokenizer.encode(turn["content"])
            )
            # we're only modifying the text in content but need to consider the formatted length
            templated_length = len(
                self.tokenizer.apply_chat_template(
                    [turn], tokenize=True, add_generation_prompt=True
                )
            )

            if total_length + templated_length > self.max_prompt_length:
                turn["content"] = self.tokenizer.decode(
                    content_token_ids[
                        : self.max_prompt_length - (total_length + templated_length)
                    ]
                )
                total_length = self.max_prompt_length
                break
            else:
                total_length += templated_length

        conversation = conversation[: (i + 1)]

        # truncate the generation if necessary
        for i, turn in enumerate(generation):
            content_token_ids = filter_out_bos_eos(
                self.tokenizer.encode(turn["content"])
            )
            # we're only modifying the text in content but need to consider the formatted length
            templated_length = len(
                self.tokenizer.apply_chat_template(
                    [turn], tokenize=True, add_generation_prompt=False
                )
            )

            if total_length + templated_length > self.max_length:
                turn["content"] = self.tokenizer.decode(
                    content_token_ids[
                        : self.max_length - (total_length + templated_length)
                    ]
                )
                total_length = self.max_length
                break
            else:
                total_length += templated_length

        generation = generation[: (i + 1)]

        tokenized_prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
        tokenized_prompt_and_generation_string = self.tokenizer.apply_chat_template(
            conversation + generation, tokenize=False, add_generation_prompt=False
        )
        tokenized_prompt_and_generation = self.tokenizer.apply_chat_template(
            conversation + generation, tokenize=True, add_generation_prompt=False
        )

        # Prepare the batch element
        batch_element = {
            "prompt": conversation,
            f"{prefix}": generation,
            "prompt_text": untruncated_prompt_string,
            "prompt_input_ids": tokenized_prompt,
            f"{prefix}_text": self.tokenizer.apply_chat_template(
                generation, tokenize=False
            ),
            f"{prefix}_combined_text": tokenized_prompt_and_generation_string,
            f"{prefix}_combined_input_ids": tokenized_prompt_and_generation,
            f"{prefix}_combined_attention_mask": [1]
            * len(tokenized_prompt_and_generation),
        }

        # Prepare labels
        tokenized_prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
        if tokenized_prompt[-1] in [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]:
            tokenized_prompt.pop()

        labels = tokenized_prompt_and_generation[:]
        labels[: len(tokenized_prompt)] = [-100] * len(tokenized_prompt)
        batch_element[f"{prefix}_labels"] = labels

        return batch_element

    def __iter__(self):
        """Create a flat version of the data and yield batches."""
        raise NotImplementedError

    def get_num_training_steps(self):
        """Get the number of training steps."""
        raise NotImplementedError


class SFTDataLoader(DataLoader):
    """
    Dataloader for supervised fine-tuning.
    """

    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())

        for prompt in prompts:
            flat_data.append(self.full_data[prompt])

        if self.num_processes == 1:  # for eval usually
            usable_size = len(flat_data)
        else:  # to avoid hanging with uneven batches
            global_batch_size = int(self.num_processes * self.microbatch_size)
            usable_size = len(flat_data) // global_batch_size * global_batch_size

        self.rng.shuffle(flat_data)
        flat_data = [
            d
            for i, d in enumerate(flat_data[:usable_size])
            if i % self.num_processes == self.process_index
        ]

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done:
                break
            self.rng.shuffle(flat_data)

            batch = []

            for example in flat_data:
                # Assuming example.prompt is now a list of conversation turns
                conversation = example.prompt
                if not isinstance(conversation[0], dict):
                    # Convert to the new format if it's not already
                    conversation = [{"role": "user", "content": conversation[0]}]
                    for i, message in enumerate(conversation[1:]):
                        role = "assistant" if i % 2 == 0 else "user"
                        conversation.append({"role": role, "content": message})

                # Get the target generation (last turn from assistant)
                target_generation = example.generations[example.sft_index]

                # Add control token if specified
                if self.control_tokens.get("chosen"):
                    target_generation = (
                        self.control_tokens["chosen"] + target_generation
                    )

                batch_element = self.tokenize_batch_element(
                    conversation, target_generation, example.truncation_mode
                )
                batch_element["original_prompt"] = example.original_prompt
                batch.append(batch_element)

                if len(batch) == self.microbatch_size:
                    example_idx += len(batch) * self.num_processes
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(
                            f"Finished generating {self.n_examples} examples on {self.split} split"
                        )
                        done = True
                        break

            if self.num_processes == 1 and batch != []:  # flush for eval, sampling
                yield self.collate(batch)
                batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

    def get_num_training_steps(self):
        """Get the number of training steps."""
        return len(self.full_data) // self.num_processes


class PairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do require pairwise preferences (e.g., DPO).
    """

    def __iter__(self):
        prompts = list(self.full_data.keys())
        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = self.rng.sample(
                    example.pairs, min(self.max_prompt_count, len(example.pairs))
                )

            for pair in example.pairs:
                flat_data.append((example, pair))

        if self.num_processes == 1:  # for eval, sampling
            usable_size = len(flat_data)
        else:  # to avoid hanging with uneven batches
            global_batch_size = int(self.num_processes * self.microbatch_size)
            usable_size = len(flat_data) // global_batch_size * global_batch_size

        self.rng.shuffle(
            flat_data
        )  # shuffle before splitting across processes, otherwise some processes will only get chosen examples
        flat_data = [
            d
            for i, d in enumerate(flat_data[:usable_size])
            if i % self.num_processes == self.process_index
        ]

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done:
                break
            self.rng.shuffle(flat_data)
            batch = []

            for example, (i, j) in flat_data:
                batch_element = {}
                batch_element.update(
                    self.tokenize_batch_element(
                        example.prompt,
                        example.generations[i],
                        example.truncation_mode,
                        prefix="chosen",
                    )
                )
                batch_element.update(
                    self.tokenize_batch_element(
                        example.prompt,
                        example.generations[j],
                        example.truncation_mode,
                        prefix="rejected",
                    )
                )
                batch_element["true_preference_labels"] = example.true_preference_labels
                batch_element["noisy_preference_labels"] = (
                    example.noisy_preference_labels
                )
                batch.append(batch_element)

                if len(batch) >= self.microbatch_size:
                    example_idx += len(batch) * self.num_processes
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(
                            f"Finished {example_idx} examples on {self.split} split"
                        )
                        done = True
                        break

            if self.num_processes == 1 and batch != []:  # flush for eval, sampling
                yield self.collate(batch)
                batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

    def get_num_training_steps(self):
        max_prompt_count = (
            min(float("inf"), self.max_prompt_count)
            if self.max_prompt_count
            else float("inf")
        )
        all_data = int(
            sum(
                min(max_prompt_count, len(example.pairs))
                for _, example in self.full_data.items()
            )
        )
        return all_data // self.num_processes
