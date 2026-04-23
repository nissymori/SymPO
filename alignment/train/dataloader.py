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
import os
from pathlib import Path
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
    if len(noisy_labels) >= 5000:
        assert (
            data_noise_ratio <= noise_ratio + 0.01
        ), f"Noise ratio is not consistent true noise ratio: {noise_ratio} data noise ratio: {data_noise_ratio} length: {len(noisy_labels)}"
        assert (
            data_noise_ratio >= noise_ratio - 0.01
        ), f"Noise ratio is not consistent true noise ratio: {noise_ratio} data noise ratio: {data_noise_ratio} length: {len(noisy_labels)}"
        
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


def get_oasst(split: str, noise_ratio: float=0.0) -> Dataset:
    """
    Load the Open Assistant dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    OASST is a dataset of ranked responses (not just pairwise), but since we are working with losses that expect paired preferences, 
    turn a ranking (a, b, c, d, e) into pairwise preferences ((a,b), (b,c), (c,d), (d,e)).
    
    Args:
        - split: one of 'test', 'train'

    Returns:   
        A Dataset instance.
    """
    rank0_print(f'Loading OASST dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('OpenAssistant/oasst1', split=('validation' if split == 'test' else 'train'))
    dataset = dataset.filter(lambda x: x['lang'] == 'en')

    message_indexed_df = pd.DataFrame(dataset).set_index('message_id')
    parent_indexed_df = pd.DataFrame(dataset).set_index('parent_id')

    def get_path_to_root(node: pd.Series):
        if node['parent_id'] is None:
            return [node]
        else:
            parent = message_indexed_df.loc[node['parent_id']]
            return [node] + get_path_to_root(parent)
    
    def build_conversation(path: List[pd.Series]):
        conversation = []
        for node in reversed(path):
            role = "user" if node['role'] == 'prompter' else "assistant"
            conversation.append({"role": role, "content": node['text']})
        return conversation

    data = Dataset('OASST')

    for row in (tqdm.tqdm(dataset, desc='Processing OASST') if on_rank0() else dataset):
        if row['rank'] == 0 or row['rank'] is None:
            continue

        try:
            sibling_df = parent_indexed_df.loc[row['parent_id']]
            next_best_sibling = sibling_df[sibling_df['rank'] == (row['rank'] - 1)].iloc[0]
            path_to_root = get_path_to_root(message_indexed_df.loc[next_best_sibling['message_id']])
        except KeyError:
            continue
        except IndexError:
            continue

        conversation = build_conversation(path_to_root[1:])  # Exclude the current message
        prompt_key = json.dumps(conversation)  # Use the conversation as the key

        data[prompt_key].prompt = conversation
        data[prompt_key].generations.append([{"role": "assistant", "content": next_best_sibling['text']}])
        data[prompt_key].generations.append([{"role": "assistant", "content": row['text']}])
        data[prompt_key].pairs.append((len(data[prompt_key].generations) - 2, len(data[prompt_key].generations) - 1))
        data[prompt_key].scores.extend([next_best_sibling['rank'], row['rank']])
        data[prompt_key].dataset_name = 'oasst'
        data[prompt_key].remove_extra_spaces()
        data[prompt_key] = flip_data(data[prompt_key], noise_ratio)

    assert_noise_ratio(data, noise_ratio)
    return data


def get_alpaca_comparison(path: str, split: str, noise_ratio: float = 0.0) -> Dataset:
    """
    Load preference comparisons from comparison_data.json and convert them to a Dataset.

    Expected JSON schema per item:
      - user_input: str                       # prompt shown to the models
      - responses_and_scores: List[Dict]      # each dict must contain:
          - response: str                     # model completion
          - score: float                      # quality score (higher is better)
          - [optional] split: {"train","test",...}

    For each prompt we form a single preference pair between the highest-scoring and
    lowest-scoring responses (skipping ties where no strict preference exists).

    Args:
        path: path to comparison_data.json
        split: split name to filter by; if items have no 'split' field, all items are used
        noise_ratio: probability of flipping each pair's preference (to inject label noise)

    Returns:
        A Dataset instance named "alpaca_comparison".
    """
    rank0_print(f"Loading Alpaca comparison data ({split} split) from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if on_rank0():
        records = tqdm.tqdm(records, desc="Processing Alpaca comparison")

    data = Dataset("alpaca_comparison")

    skipped_malformed = 0
    skipped_insufficient = 0
    skipped_ties = 0
    pair_count = 0

    for row in records:
        # Respect split only if provided in the json item
        if row.get("split", split) != split and "split" in row:
            continue

        user_input = row.get("user_input")
        responses = row.get("responses_and_scores")

        if not isinstance(user_input, str) or not isinstance(responses, list):
            skipped_malformed += 1
            continue

        valid = [
            r
            for r in responses
            if isinstance(r, dict)
            and isinstance(r.get("response"), str)
            and isinstance(r.get("score"), (int, float))
        ]

        if len(valid) < 2:
            skipped_insufficient += 1
            continue

        # Sort once to pick top/bottom efficiently
        valid.sort(key=lambda r: r["score"], reverse=True)
        best = valid[0]
        worst = min(valid, key=lambda r: r["score"])

        if float(best["score"]) <= float(worst["score"]):
            skipped_ties += 1
            continue

        key = user_input  # use the raw prompt as a stable key (consistent with other loaders)

        # Prompt as chat turns
        conversation = [{"role": "user", "content": user_input}]

        # Indices BEFORE appending the two generations
        i = data[key].num_generations()
        j = i + 1

        # Highest-scoring response is treated as the preferred completion.
        data[key].prompt = conversation
        data[key].generations.append(
            [{"role": "assistant", "content": best["response"]}]
        )
        data[key].generations.append(
            [{"role": "assistant", "content": worst["response"]}]
        )
        data[key].scores.extend([float(best["score"]), float(worst["score"])])
        data[key].pairs.append((i, j))
        data[key].sft_index = i  # chosen response as SFT target
        data[key].dataset_name = data.name
        data[key].original_prompt = user_input
        data[key].truncation_mode = "keep_start"

        # Normalize spacing
        data[key].remove_extra_spaces()

        # Inject label noise (flip pair with prob=noise_ratio) and record labels
        data[key] = flip_data(data[key], noise_ratio)
        pair_count += 1

    if on_rank0():
        if skipped_malformed:
            rank0_print(f"Skipped {skipped_malformed} malformed items in {path}.")
        if skipped_insufficient:
            rank0_print(
                f"Skipped {skipped_insufficient} items with fewer than two scored responses."
            )
        if skipped_ties:
            rank0_print(
                f"Skipped {skipped_ties} items where top and bottom scores were tied."
            )

    if pair_count == 0:
        raise ValueError(
            f"No Alpaca comparison pairs could be constructed from {path}; check the data file."
        )

    # Sanity-check the achieved noise ratio (no-op if noise_ratio==0)
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

        default_comparison_path = Path(__file__).resolve().parent / "data" / "comparison_data_v2.json"
        supplied_path = kwargs.get("data_path")
        if supplied_path is None:
            resolved_data_path = default_comparison_path
        else:
            supplied_path = Path(os.path.expanduser(str(supplied_path)))
            if supplied_path.is_absolute():
                resolved_data_path = supplied_path
            else:
                module_candidate = Path(__file__).resolve().parent / supplied_path
                cwd_candidate = Path.cwd() / supplied_path
                if module_candidate.exists():
                    resolved_data_path = module_candidate
                elif cwd_candidate.exists():
                    resolved_data_path = cwd_candidate
                else:
                    resolved_data_path = module_candidate
        self.data_path = str(resolved_data_path)

        assert (
            n_epochs is not None or n_examples is not None
        ), "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples

        self.full_data = {}  # a dict of Examples

        for name in dataset_names:
            if f"get_{name}" in globals():
                if name == "alpaca_comparison":
                    current_path = self.data_path
                    dataset = globals()[f"get_{name}"](current_path, split, noise_ratio=self.noise_ratio)
                else:
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
