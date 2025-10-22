# Copyright 2025 Tencent Inc. All Rights Reserved.
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

import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import ray
from fastchat.llm_judge.common import load_questions
from transformers import AutoTokenizer

from .generate_baseline_answer import get_model_answers as get_baseline_answers
from .generate_eagle_answer import get_model_answers as get_eagle_answers


class BenchmarkMode(Enum):
    """Benchmark execution modes"""

    EAGLE = "eagle"
    BASELINE = "baseline"
    BOTH = "both"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""

    # Model paths
    base_model_path: str
    eagle_model_path: str
    model_id: str

    # Benchmark settings
    bench_name: str = "mt_bench"
    question_begin: Optional[int] = None
    question_end: Optional[int] = None

    # Generation parameters
    num_choices: int = 1
    temperature: float = 1.0
    max_new_token: int = 1024
    total_token: int = 60
    depth: int = 5
    top_k: int = 10

    # Hardware settings
    num_gpus_per_model: int = 1
    num_gpus_total: int = 1
    max_gpu_memory: Optional[str] = None

    # Output settings
    output_dir: Optional[str] = None
    seed: int = 42

    # Analysis settings
    calculate_metrics: bool = True

    # SpecExit
    early_stop_method: Optional[str] = None


class BenchmarkEngine:
    """Core benchmark engine for speculative decoding evaluation"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self._setup_paths()

    def _setup_paths(self):
        """Setup output paths for benchmark results"""
        if not self.config.output_dir:
            script_dir = os.path.dirname(__file__)
            parent_dir = os.path.dirname(os.path.dirname(script_dir))
            self.config.output_dir = f"{parent_dir}/output/{self.config.bench_name}"

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup result file paths
        model_id_temp = f"{self.config.model_id}-temperature-{self.config.temperature}"
        self.eagle_file = os.path.join(
            self.config.output_dir, f"{model_id_temp}-eagle.jsonl"
        )
        self.baseline_file = os.path.join(
            self.config.output_dir, f"{model_id_temp}-baseline.jsonl"
        )
        self.analysis_file = os.path.join(
            self.config.output_dir, f"{model_id_temp}-analysis.json"
        )

    def run_benchmark(self, mode: BenchmarkMode = BenchmarkMode.BOTH) -> Dict[str, Any]:
        """
        Run benchmark with specified mode

        Args:
            mode: Benchmark execution mode (EAGLE/BASELINE/BOTH)

        Returns:
            Dictionary containing benchmark results and analysis
        """
        print(f"Starting benchmark in {mode.value} mode...")

        # Initialize Ray if needed
        use_ray = self.config.num_gpus_total // self.config.num_gpus_per_model > 1
        if use_ray:
            ray.init()

        try:
            if mode in [BenchmarkMode.EAGLE, BenchmarkMode.BOTH]:
                print("Running Eagle benchmark...")
                self._run_eagle_benchmark()

            if mode in [BenchmarkMode.BASELINE, BenchmarkMode.BOTH]:
                print("Running baseline benchmark...")
                self._run_baseline_benchmark()

            # Calculate metrics if requested
            if self.config.calculate_metrics and mode == BenchmarkMode.BOTH:
                self.results.update(self._calculate_metrics())

            # Save analysis results
            self._save_analysis()

        finally:
            if use_ray:
                ray.shutdown()

        return self.results

    def _run_eagle_benchmark(self):
        """Run Eagle speculative decoding benchmark"""
        args = self._create_args_namespace("eagle")

        questions = load_questions(
            self._get_question_file_path(),
            self.config.question_begin,
            self.config.question_end,
        )

        use_ray = self.config.num_gpus_total // self.config.num_gpus_per_model > 1
        get_answers_func = (
            ray.remote(num_gpus=self.config.num_gpus_per_model)(
                get_eagle_answers
            ).remote
            if use_ray
            else get_eagle_answers
        )

        chunk_size = len(questions) // (
            self.config.num_gpus_total // self.config.num_gpus_per_model
        )
        ans_handles = [
            get_answers_func(
                f"{self.config.model_id}-temperature-{self.config.temperature}",
                questions[i : i + chunk_size],
                self.eagle_file,
                self.config.num_choices,
                self.config.temperature,
                args,
            )
            for i in range(0, len(questions), chunk_size)
        ]

        if use_ray:
            ray.get(ans_handles)

        self._reorg_answer_file(self.eagle_file)
        self.results["eagle_file"] = self.eagle_file

    def _run_baseline_benchmark(self):
        """Run baseline benchmark"""
        args = self._create_args_namespace("baseline")

        questions = load_questions(
            self._get_question_file_path(),
            self.config.question_begin,
            self.config.question_end,
        )

        use_ray = self.config.num_gpus_total // self.config.num_gpus_per_model > 1
        get_answers_func = (
            ray.remote(num_gpus=self.config.num_gpus_per_model)(
                get_baseline_answers
            ).remote
            if use_ray
            else get_baseline_answers
        )

        chunk_size = len(questions) // (
            self.config.num_gpus_total // self.config.num_gpus_per_model
        )
        ans_handles = [
            get_answers_func(
                f"{self.config.model_id}-temperature-{self.config.temperature}",
                questions[i : i + chunk_size],
                self.baseline_file,
                self.config.num_choices,
                self.config.temperature,
                args,
            )
            for i in range(0, len(questions), chunk_size)
        ]

        if use_ray:
            ray.get(ans_handles)

        self._reorg_answer_file(self.baseline_file)
        self.results["baseline_file"] = self.baseline_file

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate acceptance length and speedup ratio"""
        metrics = {}

        # Calculate acceptance length from Eagle results
        if os.path.exists(self.eagle_file):
            metrics["acceptance_length"] = self._calculate_acceptance_length(
                self.eagle_file
            )

        # Calculate speedup ratio if both files exist
        if os.path.exists(self.eagle_file) and os.path.exists(self.baseline_file):
            metrics["speedup_ratio"] = self._calculate_speedup_ratio(
                self.config.base_model_path, self.baseline_file, self.eagle_file
            )

        return metrics

    def _calculate_acceptance_length(self, input_file: str) -> float:
        """
        Calculate average acceptance length from benchmark results.

        Args:
            input_file: Path to JSONL file containing benchmark results

        Returns:
            Average acceptance length
        """
        with open(input_file, "r") as f:
            lines = f.readlines()

        print(f"Number of samples: {len(lines)}")
        avg_accept_length = 0.0

        for line in lines:
            data = json.loads(line)
            accept_lengths = data["choices"][0]["accept_length"]
            avg_accept_length += sum(accept_lengths) / len(accept_lengths) + 1

        avg_accept_length /= len(lines)
        return avg_accept_length

    def _calculate_speedup_ratio(
        self, model_path: str, baseline_json: str, eagle_json: str
    ) -> float:
        """
        Calculate speedup ratio between baseline and speculative decoding.

        Args:
            model_path: Path to HuggingFace model for tokenization
            baseline_json: Path to baseline benchmark results
            eagle_json: Path to speculative decoding benchmark results

        Returns:
            Speedup ratio (eagle speed / baseline speed)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Process speculative decoding results
        eagle_speeds = []
        with open(eagle_json, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                tokens = sum(data["choices"][0]["new_tokens"])
                times = sum(data["choices"][0]["wall_time"])
                eagle_speeds.append(tokens / times)

        # Process baseline results
        baseline_speeds = []
        with open(baseline_json, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                answers = data["choices"][0]["turns"]
                tokens = sum(len(tokenizer(ans).input_ids) - 1 for ans in answers)
                times = sum(data["choices"][0]["wall_time"])
                baseline_speeds.append(tokens / times)

        return np.array(eagle_speeds).mean() / np.array(baseline_speeds).mean()

    def _create_args_namespace(self, mode: str) -> argparse.Namespace:
        """Create argparse.Namespace from config for compatibility"""
        args = argparse.Namespace()

        # Model paths
        args.base_model_path = self.config.base_model_path
        args.eagle_model_path = self.config.eagle_model_path
        args.model_id = self.config.model_id

        # Benchmark settings
        args.bench_name = self.config.bench_name
        args.question_begin = self.config.question_begin
        args.question_end = self.config.question_end

        # Generation parameters
        args.num_choices = self.config.num_choices
        args.temperature = self.config.temperature
        args.max_new_token = self.config.max_new_token
        args.total_token = self.config.total_token
        args.depth = self.config.depth
        args.top_k = self.config.top_k

        # Hardware settings
        args.num_gpus_per_model = self.config.num_gpus_per_model
        args.num_gpus_total = self.config.num_gpus_total
        args.max_gpu_memory = self.config.max_gpu_memory

        # Output settings
        if mode == "eagle":
            args.answer_file = self.eagle_file
        else:
            args.answer_file = self.baseline_file

        args.seed = self.config.seed
        args.load_in_8bit = False

        args.early_stop_method = self.config.early_stop_method

        return args

    def _get_question_file_path(self) -> str:
        """Get question file path"""
        current_file = os.path.abspath(__file__)
        project_root = current_file.split("/AngelSlim/")[0] + "/AngelSlim"
        return os.path.join(
            project_root, "dataset", self.config.bench_name, "question.jsonl"
        )

    def _reorg_answer_file(self, answer_file: str):
        """Sort answers by question id and remove duplicates"""
        answers = {}
        with open(answer_file, "r") as fin:
            for line in fin:
                qid = json.loads(line)["question_id"]
                answers[qid] = line

        with open(answer_file, "w") as fout:
            for qid in sorted(answers.keys()):
                fout.write(answers[qid])

    def _save_analysis(self):
        """Save analysis results to file"""
        with open(self.analysis_file, "w") as f:
            json.dump(self.results, f, indent=4)

    def get_performance_summary(self) -> str:
        """Generate a performance summary report"""
        if not self.results:
            return "No benchmark results available."

        summary = [
            "=== Speculative Decoding "
            f"{self.config.bench_name.upper()} Benchmark Results ===\n"
        ]

        if "acceptance_length" in self.results:
            summary.append(
                f"Average Acceptance Length: {self.results['acceptance_length']:.2f}"
            )

        if "speedup_ratio" in self.results:
            summary.append(f"Speedup Ratio: {self.results['speedup_ratio']:.2f}x")

        if "eagle_file" in self.results:
            summary.append(f"Eagle Results: {self.results['eagle_file']}")

        if "baseline_file" in self.results:
            summary.append(f"Baseline Results: {self.results['baseline_file']}")

        summary.append(f"Analysis Report: {self.analysis_file}")

        return "\n".join(summary)
