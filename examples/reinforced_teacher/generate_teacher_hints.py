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
Generate frozen hints from teacher model for evaluation.
This is used for the frozen hints baseline.
"""

import argparse
import os
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_hint(model, tokenizer, question: str, device="cuda") -> str:
    """Generate a hint for the given question using the teacher model."""
    prompt = f"""You are a helpful teacher. Provide a brief hint (1-2 sentences) to help solve this math problem. Do not give the full solution.

Problem: {question}

Hint:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    hint = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return hint.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", required=True, help="Path to teacher model")
    parser.add_argument("--test_file", required=True, help="Input test parquet file")
    parser.add_argument("--output_file", required=True, help="Output parquet file with hints")
    parser.add_argument("--num_hints", type=int, default=1, help="Number of hints per question")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    print(f"Loading teacher model: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Loading test data from: {args.test_file}")
    df = pd.read_parquet(args.test_file)

    # Generate hints for each question
    hints_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating hints"):
        question = row["prompt"][0]["content"]

        # Generate hint(s)
        hint = generate_hint(model, tokenizer, question, device=args.device)

        # Create modified prompt with hint
        prompt_with_hint = [
            {
                "role": "user",
                "content": question + f"\n\nHint: {hint}"
            }
        ]

        # Create new row with hint
        new_row = row.copy()
        new_row["prompt"] = prompt_with_hint
        new_row["extra_info"]["teacher_hint"] = hint
        hints_data.append(new_row)

    # Save to parquet
    output_df = pd.DataFrame(hints_data)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_df.to_parquet(args.output_file)

    print(f"Saved {len(output_df)} examples with hints to: {args.output_file}")
