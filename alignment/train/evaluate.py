import argparse
import json
from typing import Dict, List, TextIO, Type, Union

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


class StreamingJSONWriter:
    """Writes JSON arrays to a file in a streaming fashion."""

    def __init__(self, file: TextIO):
        self.file = file
        self.is_first = True
        self.file.write("[\n")

    def write_item(self, item: Dict):
        """Write a single item to the JSON array."""
        if not self.is_first:
            self.file.write(",\n")
        json.dump(item, self.file, indent=2)
        self.is_first = False
        # Flush after each write to ensure immediate disk writing
        self.file.flush()

    def close(self):
        """Close the JSON array and the file."""
        self.file.write("\n]")
        self.file.flush()


prompt_templates = {}
prompt_templates[
    "default"
] = """
For the following query to a chatbot, determine which response is more helpful.

**Query:** {query}

**Response A:** {response_A}

**Response B:** {response_B}

FIRST, provide a one-sentence comparison of the two responses, explaining which response is more helpful by indicating that it offers accurate information. SECOND, on a new line, state only "A" or "B" to indicate which response is more helpful. 

Use the following format:

Comparison: <one-sentence comparison and explanation>

More helpful: <"A" or "B">
"""


def get_gpt4_judgement(example, args):
    query = example["prompt"][0]["content"]
    prompt = prompt_templates[args.task].format(
        query=query, response_A=example["ref_output"], response_B=example["output"]
    )
    completion = client.chat.completions.create(
        seed=args.seed,
        model=args.evaluator,
        messages=[{"role": "user", "content": prompt}],
    )
    judgement = completion.choices[0].message.content
    return judgement


def win_check(example, task, args=None):
    if task in ["default"]:
        if "judgement" not in example:
            judgement = get_gpt4_judgement(example, args)
        else:
            judgement = example["judgement"]
        if "More helpful" in judgement:
            if "More helpful: B" in judgement or 'More helpful: "B"' in judgement:
                return "win", judgement
            else:
                return "lose", judgement
        else:
            return None, judgement
    else:
        raise Exception("Task not found.")


def main(args):

    with open(args.input_file, "r") as file:
        data = json.load(file)

    output_file = "/".join(
        args.input_file.split("/")[:-1]
        + [f"judgement_{args.input_file.split('/')[-1]}"]
    )

    all_win_lose = []
    with open(output_file, "w") as f:
        writer = StreamingJSONWriter(f)

        for i, example in enumerate(data):
            win_lose, judgement = win_check(example, args.task, args)
            all_win_lose.append(win_lose)

            example["judgement"] = judgement
            writer.write_item(example)

            print(
                f"Evaluated examples {i+1}, N of wins = {all_win_lose.count('win')}/{len(data)}",
                end="\r",
            )

        writer.close()

    wr = all_win_lose.count("win") / (
        all_win_lose.count("win") + all_win_lose.count("lose")
    )

    d = {
        "win rate": [wr],
        "N of wins": [all_win_lose.count("win")],
        "N of loses": [all_win_lose.count("lose")],
        "N of null": [
            len(all_win_lose) - (all_win_lose.count("win") + all_win_lose.count("lose"))
        ],
    }

    df = pd.DataFrame(data=d)

    df.to_csv(
        "/".join(
            args.input_file.split("/")[:-1]
            + [f"stats_{args.input_file.split('/')[-1].split('.json')[0]}.csv"]
        ),
        index=False,
    )

    print(f"WR = {wr:.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the JSON file containing sampled outputs",
    )
    parser.add_argument(
        "--task", type=str, default="default", choices=["default"], help="Task"
    )
    parser.add_argument("--seed", type=int, default=0, help="OpenAI seed")
    parser.add_argument(
        "--evaluator",
        type=str,
        default="gpt-4o-mini",
        help="Model evaluator, e.g., gpt-4o-mini",
    )

    args = parser.parse_args()

    client = OpenAI()

    main(args)
