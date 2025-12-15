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
Custom reward function for ReinforcedTeacher GRPO.

The teacher model generates hints for the student model.
- If student with hints gets correct answer: teacher gets reward +1
- If student with hints gets incorrect answer: teacher gets reward 0
- Student always gets shaped reward based on correctness
"""

import re
from typing import Dict, Any


def extract_hint_from_response(response_str: str) -> str:
    """
    Extract the hint portion from teacher's response.
    Assumes format: "Hint: <hint text>"
    """
    hint_match = re.search(r"Hint:\s*(.*?)(?:\n|$)", response_str, re.IGNORECASE | re.DOTALL)
    if hint_match:
        return hint_match.group(1).strip()
    return response_str.strip()


def extract_solution(solution_str: str) -> str:
    """Extract numerical answer from solution string."""
    # Look for #### format (GSM8K style)
    solutions = re.findall(r"#### (\-?[0-9\.\\,]+)", solution_str)
    if len(solutions) > 0:
        return solutions[-1].replace(",", "").replace("$", "")

    # Fallback: look for last number in response (last 300 chars)
    if len(solution_str) > 300:
        solution_str = solution_str[-300:]

    numbers = re.findall(r"(\-?[0-9\.\\,]+)", solution_str)
    if len(numbers) > 0:
        # Get last valid number
        for num in reversed(numbers):
            if num not in ["", "."]:
                return num.replace(",", "").replace("$", "")

    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """
    Compute reward score for ReinforcedTeacher setup.

    Args:
        data_source: The source of the data (e.g., "math_small")
        solution_str: The generated response from the model
        ground_truth: The correct answer
        extra_info: Additional info including:
            - is_teacher: bool, whether this is teacher model generating hints
            - student_correct: bool, whether student got correct answer (only for teacher)
            - hint: str, the hint provided by teacher (only for student)
    """
    extra_info = extra_info or {}

    # Check if this is a teacher model or student model
    is_teacher = extra_info.get("is_teacher", False)

    if is_teacher:
        # Teacher model: reward based on whether student succeeded with the hint
        student_correct = extra_info.get("student_correct", False)
        if student_correct:
            return 1.0  # Positive reward: hint helped student get correct answer
        else:
            return 0.0  # Zero reward: hint didn't help student
    else:
        # Student model: standard correctness-based reward
        predicted_answer = extract_solution(solution_str)

        if predicted_answer is None:
            return 0.0  # No valid answer found

        if predicted_answer == ground_truth:
            return 1.0  # Correct answer
        else:
            return 0.0  # Incorrect answer


def compute_score_baseline(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """
    Baseline reward function for evaluation without teacher hints.
    Just checks if the answer is correct.
    """
    predicted_answer = extract_solution(solution_str)

    if predicted_answer is None:
        return 0.0

    if predicted_answer == ground_truth:
        return 1.0
    else:
        return 0.0


def compute_score_frozen_hints(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """
    Reward function for evaluation with frozen teacher hints.
    Same as baseline but expects hints in the prompt.
    """
    return compute_score_baseline(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs
    )
