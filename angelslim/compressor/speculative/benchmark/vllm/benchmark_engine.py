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

import ray
from fastchat.llm_judge.common import load_questions

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
    batch_size: int = 200
    temperature: float = 0.0
    max_new_token: int = 1024
    depth: int = 5
    top_k: int = 0
    top_p: float = 1.0

    # Hardware settings
    num_gpus_per_model: int = 1
    num_gpus_total: int = 1
    max_gpu_memory: Optional[str] = None

    # Output settings
    output_dir: Optional[str] = None
    seed: int = 42

    # Analysis settings
    calculate_metrics: bool = True


class BenchmarkEngine:
    """Core benchmark engine for speculative decoding evaluation with vLLM"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self._setup_paths()

    def _setup_paths(self):
        """Setup output paths for benchmark results"""
        if not self.config.output_dir:
            script_dir = os.path.dirname(__file__)
            parent_dir = os.path.dirname(os.path.dirname(script_dir))
            self.config.output_dir = os.path.join(
                parent_dir, "output", self.config.bench_name
            )

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
        print(f"Starting vLLM benchmark in {mode.value} mode...")

        # Initialize Ray if needed
        use_ray = self.config.num_gpus_total // self.config.num_gpus_per_model > 1
        if use_ray:
            ray.init()

        try:
            if mode == BenchmarkMode.EAGLE or mode == BenchmarkMode.BOTH:
                print("Running Eagle speculative decoding benchmark...")
                self._run_eagle_benchmark()

            if mode == BenchmarkMode.BASELINE or mode == BenchmarkMode.BOTH:
                print("Running baseline benchmark...")
                self._run_baseline_benchmark()

            if self.config.calculate_metrics:
                print("Calculating metrics...")
                metrics = self._calculate_metrics()
                self.results["metrics"] = metrics
                self._save_analysis()

        finally:
            if use_ray:
                ray.shutdown()

        return self.results

    def _run_eagle_benchmark(self):
        """Run Eagle speculative decoding benchmark with vLLM"""
        args = self._create_args_namespace("eagle")

        question_file = self._get_question_file_path()
        questions = load_questions(
            question_file,
            self.config.question_begin,
            self.config.question_end,
        )
        print(f"Total {len(questions)} questions from file {question_file}")

        use_ray = (self.config.num_gpus_total // self.config.num_gpus_per_model) > 1
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
            results = ray.get(ans_handles)
            avg_accept_lengths = [r for r in results if r is not None]
            print(f"Avg acceptance lengths: {avg_accept_lengths}")
            if avg_accept_lengths:
                self.results["average_acceptance_length"] = sum(
                    avg_accept_lengths
                ) / len(avg_accept_lengths)
        else:
            # For a single process, there's only one result
            result = ans_handles[0]
            print(f"Acceptance length from single process: {result}")
            if result is not None:
                self.results["average_acceptance_length"] = result

        self._reorg_answer_file(self.eagle_file)
        self.results["eagle_file"] = self.eagle_file

    def _run_baseline_benchmark(self):
        """Run baseline benchmark with vLLM"""
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

    def _calculate_acceptance_length(self, input_file: str) -> float:
        """
        Calculate average acceptance length from benchmark results.

        Args:
            input_file: Path to JSONL file containing benchmark results

        Returns:
            Average acceptance length
        """
        if "average_acceptance_length" in self.results:
            avg_accept = self.results["average_acceptance_length"]
            print(f"Average acceptance length: {avg_accept:.4f}")
            return avg_accept
        return 0.0

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate acceptance length and speedup ratio"""
        metrics = {}

        # Use the acceptance length calculated during the benchmark run
        if "average_acceptance_length" in self.results:
            avg_accept = self.results["average_acceptance_length"]
            metrics["average_acceptance_length"] = avg_accept
            print(f"Average acceptance length: {avg_accept:.4f}")

        # Calculate speedup ratio if both files exist
        if os.path.exists(self.eagle_file) and os.path.exists(self.baseline_file):
            speedup = self._calculate_speedup_ratio(
                self.config.base_model_path, self.baseline_file, self.eagle_file
            )
            metrics["speedup_ratio"] = speedup
            print(f"Speedup ratio: {speedup:.4f}x")

        return metrics

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
        # Process speculative decoding results
        eagle_speeds = []
        with open(eagle_json, "r") as f:
            for line in f:
                data = json.loads(line)
                for choice in data["choices"]:
                    total_tokens = sum(choice["new_tokens"])
                    total_time = sum(choice["wall_time"])
                    if total_time > 0:
                        eagle_speeds.append(total_tokens / total_time)

        # Process baseline results
        baseline_speeds = []
        with open(baseline_json, "r") as f:
            for line in f:
                data = json.loads(line)
                for choice in data["choices"]:
                    total_tokens = sum(choice["new_tokens"])
                    total_time = sum(choice["wall_time"])
                    if total_time > 0:
                        baseline_speeds.append(total_tokens / total_time)

        avg_eagle_speed = sum(eagle_speeds) / len(eagle_speeds) if eagle_speeds else 0
        avg_baseline_speed = (
            sum(baseline_speeds) / len(baseline_speeds) if baseline_speeds else 0
        )

        speedup_ratio = (
            avg_eagle_speed / avg_baseline_speed if avg_baseline_speed > 0 else 0
        )
        return speedup_ratio

    def _create_args_namespace(self, mode: str) -> argparse.Namespace:
        """Create argument namespace for generation functions"""
        args = argparse.Namespace(
            base_model_path=self.config.base_model_path,
            eagle_model_path=self.config.eagle_model_path,
            model_id=self.config.model_id,
            bench_name=self.config.bench_name,
            question_begin=self.config.question_begin,
            question_end=self.config.question_end,
            answer_file=self.eagle_file if mode == "eagle" else self.baseline_file,
            max_new_token=self.config.max_new_token,
            depth=self.config.depth,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            num_choices=self.config.num_choices,
            num_gpus_per_model=self.config.num_gpus_per_model,
            num_gpus_total=self.config.num_gpus_total,
            batch_size=self.config.batch_size,
            temperature=self.config.temperature,
            seed=self.config.seed,
        )
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
            json.dump(self.results, f, indent=2)

    def get_performance_summary(self) -> str:
        """Get a formatted performance summary"""
        summary = ["=" * 60, "vLLM Benchmark Performance Summary", "=" * 60, ""]

        if "eagle_file" in self.results:
            summary.append(f"Eagle results: {self.results['eagle_file']}")

        if "baseline_file" in self.results:
            summary.append(f"Baseline results: {self.results['baseline_file']}")

        if "metrics" in self.results:
            summary.append("\nMetrics:")
            for key, value in self.results["metrics"].items():
                summary.append(f"  {key}: {value:.4f}")

        summary.append("\n" + "=" * 60)
        return "\n".join(summary)
