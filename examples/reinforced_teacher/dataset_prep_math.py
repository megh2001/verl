# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess math dataset for ReinforcedTeacher GRPO training.
Creates a simple small math dataset for proof of concept.
"""

import argparse
import os
import json

import datasets
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """Extract numerical solution from GSM8K style answer."""
    import re
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/math_small",
                        help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--num_train", type=int, default=500,
                        help="Number of training examples")
    parser.add_argument("--num_test", type=int, default=100,
                        help="Number of test examples")

    args = parser.parse_args()

    # Load GSM8K as our base math dataset
    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset(data_source, "main")

    # Take subset for POC
    train_dataset = dataset["train"].select(range(min(args.num_train, len(dataset["train"]))))
    test_dataset = dataset["test"].select(range(min(args.num_test, len(dataset["test"]))))

    instruction = "Solve the following math problem step by step."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = instruction + "\n\n" + question_raw

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)

            data = {
                "data_source": "math_small",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "reinforced_teacher",  # Custom reward style
                    "ground_truth": solution
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Saved {len(train_dataset)} training examples to {local_save_dir}/train.parquet")
    print(f"Saved {len(test_dataset)} test examples to {local_save_dir}/test.parquet")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
