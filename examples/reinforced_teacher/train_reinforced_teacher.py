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
ReinforcedTeacher GRPO Training Loop

Architecture:
1. Teacher model (7B) generates hints for each math question
2. Student model (2.5B) attempts to solve with the hints
3. Rewards:
   - Teacher: +1 if student correct with hint, 0 if incorrect
   - Student: +1 if correct answer, 0 if incorrect
4. Train teacher with GRPO to generate better hints
"""

import os
import argparse
from typing import Dict, List, Any
import torch
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


class ReinforcedTeacherTrainer:
    """Manages the two-stage training loop for ReinforcedTeacher."""

    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        data_path: str,
        output_dir: str,
        device: str = "cuda"
    ):
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = device

        os.makedirs(output_dir, exist_ok=True)

    def generate_hints_from_teacher(
        self,
        questions: List[str],
        n_hints: int = 5
    ) -> List[List[str]]:
        """
        Generate multiple hints per question using teacher model via vLLM rollout.
        Returns list of hint lists (one list per question).
        """
        # This will be called via the GRPO rollout mechanism
        # For now, this is a placeholder that shows the interface
        pass

    def generate_student_solutions(
        self,
        questions: List[str],
        hints: List[str]
    ) -> List[str]:
        """
        Generate solutions from student given questions and hints.
        """
        pass

    def compute_rewards(
        self,
        student_solutions: List[str],
        ground_truths: List[str],
        hints: List[str]
    ) -> tuple[List[float], List[float]]:
        """
        Compute rewards for both teacher and student.

        Returns:
            teacher_rewards: List of rewards for teacher (based on student correctness)
            student_rewards: List of rewards for student (based on own correctness)
        """
        from reward_function import extract_solution

        student_rewards = []
        teacher_rewards = []

        for solution, truth in zip(student_solutions, ground_truths):
            predicted = extract_solution(solution)
            is_correct = (predicted == truth)

            # Student reward: 1.0 if correct, 0.0 otherwise
            student_rewards.append(1.0 if is_correct else 0.0)

            # Teacher reward: 1.0 if student correct (hint helped), 0.0 otherwise
            teacher_rewards.append(1.0 if is_correct else 0.0)

        return teacher_rewards, student_rewards

    def train_epoch(self, epoch: int):
        """Run one epoch of ReinforcedTeacher training."""
        print(f"\n=== Epoch {epoch} ===")

        # Load data
        df = pd.read_parquet(self.data_path)

        # This is a simplified pseudocode showing the training flow
        # The actual training uses VERL's GRPO trainer which is called via shell script

        print("Training flow:")
        print("1. Teacher generates hints (via GRPO rollout)")
        print("2. Student generates solutions with hints")
        print("3. Compute rewards based on student correctness")
        print("4. Update teacher policy with GRPO")
        print("5. (Optional) Update student policy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./outputs/reinforced_teacher")
    parser.add_argument("--n_epochs", type=int, default=10)

    args = parser.parse_args()

    trainer = ReinforcedTeacherTrainer(
        teacher_model_path=args.teacher_model,
        student_model_path=args.student_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

    for epoch in range(args.n_epochs):
        trainer.train_epoch(epoch)


if __name__ == "__main__":
    print("=" * 80)
    print("ReinforcedTeacher GRPO Training")
    print("=" * 80)
    print()
    print("NOTE: This script is a reference implementation showing the training flow.")
    print("The actual training is orchestrated via the shell scripts which call")
    print("verl.trainer.main_ppo with custom configurations.")
    print()
    print("Please use run_reinforced_teacher_grpo.sh to start training.")
    print("=" * 80)
