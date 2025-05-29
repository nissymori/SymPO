import argparse
import inspect
import os
import re

import train.dataloader as dataloader_module
from train.dataloader import SFTDataLoader
from train.utils import set_offline_if_needed
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

from .utils import StreamingJSONWriter


def get_available_datasets():
    """Get list of available datasets by finding all get_* functions in dataloader.py"""
    return [
        name[4:]
        for name, _ in inspect.getmembers(dataloader_module, inspect.isfunction)
        if name.startswith("get_")
    ]


def validate_datasets(datasets):
    """Validate that all requested datasets have corresponding get_* functions"""
    available_datasets = get_available_datasets()
    invalid_datasets = [d for d in datasets if d not in available_datasets]

    if invalid_datasets:
        available_str = "\n- ".join(available_datasets)
        raise ValueError(
            f"The following datasets are not available: {invalid_datasets}\n"
            f"Available datasets must have a corresponding get_* function in dataloader.py.\n"
            f"Currently available datasets are:\n- {available_str}"
        )


def main(args):
    # validate_datasets(args.datasets) # skip
    set_offline_if_needed()

    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpu_count)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.chat_template = open("config/template.jinja").read()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create output folder if not exist
    output_folder = os.path.join("/".join(args.output_file.split("/")[:-1]))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the output file and create a streaming writer
    with open(args.output_file, "w") as f:
        writer = StreamingJSONWriter(f)

        # Process each dataset
        for dataset in args.datasets:

            for seed in args.seeds:
                prompt_idx = 0
                print(f"\nProcessing dataset: {dataset} at seed {seed}")

                # Initialize the SFTDataLoader
                dataloader = SFTDataLoader(
                    dataset_names=[dataset],
                    tokenizer=tokenizer,
                    split=args.split,
                    max_prompt_length=args.max_prompt_length,
                    n_examples=args.n_examples,
                    seed=seed,
                    microbatch_size=args.batch_size,
                )

                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    stop=[args.stop_token],
                    n=args.num_samples_per_prompt,
                    seed=seed,
                )

                # Process the dataset in batches
                for batch in dataloader:
                    # prompt_text has already had the chat template applied
                    responses = llm.generate(batch["prompt_text"], sampling_params)

                    # Process and write each output
                    for prompt, response, ref_response in zip(
                        batch["prompt"], responses, batch["target_text"]
                    ):
                        for sample_idx, sample in enumerate(response.outputs):
                            output = {
                                "output": re.sub(
                                    r"<?\|(im_start|im_end)\|>?",
                                    "",
                                    sample.text.strip(),
                                ),
                                "generator": args.model_path,
                                "dataset": f"{dataset}_{args.split}",
                                "prompt_id": prompt_idx,
                                "sample_id": sample_idx,
                                "type": "sample",
                                "ref_output": re.sub(
                                    r"<?\|(im_start|im_end)\|>?",
                                    "",
                                    ref_response.replace("assistant\n", "").strip(),
                                ),
                                "sampling_seed": seed,
                            }

                            # for eval with alpacaeval
                            if args.mode == "alpacaeval":
                                output["instruction"] = prompt[0]["content"]
                            else:
                                output["prompt"] = prompt

                            writer.write_item(output)

                        prompt_idx += 1

        writer.close()

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from a local model using vllm for AlpacaEval"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the local model folder or the Huggingface repo",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["alpacaeval"],
        help="List of datasets to sample from (space-separated)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs.json",
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--gpu_count", type=int, default=1, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum length of prompt (in tokens)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for processing datasets"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="List of random seeds for reproducibility (space-separated)",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use (train/test)"
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=1,
        help="Number of samples to generate per input",
    )
    parser.add_argument(
        "--stop_token", type=str, default="<|im_end|>", help="Stop token"
    )
    parser.add_argument("--mode", type=str, default="alpacaeval", help="mode")
    parser.add_argument(
        "--n_examples", type=int, default=1024, help="Number of examples"
    )

    args = parser.parse_args()
    main(args)
