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

"""Setup for pip package."""
import os
import subprocess

from setuptools import find_packages, setup


def _read_pkg_info_version():
    """Reuse PKG-INFO's resolved version when building from an sdist.

    An sdist ships a top-level ``PKG-INFO`` whose ``Version:`` line already
    carries the fully-resolved version (including the ``+cuXXX.torchY`` local
    segment). When pip's PEP 517 backend re-runs setup.py from the extracted
    sdist, reuse that verbatim — recomputing would (a) call ``git`` in a non-git
    tree and (b) re-append the local segment, producing an invalid double-local
    version. A fresh git checkout has no top-level PKG-INFO, so this returns None
    there and the git+torch path below runs.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_info = os.path.join(here, "PKG-INFO")
    if os.path.isfile(pkg_info):
        try:
            with open(pkg_info) as f:
                for line in f:
                    if line.startswith("Version:"):
                        v = line.split(":", 1)[1].strip()
                        if v:
                            return v
        except OSError:
            pass
    return None


def _git_base_version():
    """``0.0.0_dev`` on a checkout that has a ``main`` branch, else the latest
    tag. Returns None when not in a git tree (an extracted sdist) so the caller
    falls back to a static base — a bare ``git branch`` there returns
    ``fatal: not a git repository...``, which must NOT become the version string
    (it raises packaging.version.InvalidVersion and breaks `pip install`).
    """
    try:
        branch = subprocess.run(["git", "branch"], capture_output=True, text=True)
        if branch.returncode != 0:
            return None
        if "main" in branch.stdout:
            return "0.0.0_dev"
        tags = subprocess.run(["git", "tag"], capture_output=True, text=True)
        if tags.returncode != 0:
            return None
        tag_list = [t for t in tags.stdout.split("\n") if t.strip()]
        return tag_list[-1] if tag_list else "0.0.0_dev"
    except (OSError, subprocess.SubprocessError):
        # git binary missing, etc. — degrade to the static base, never crash.
        return None


# Static fallback so a non-git, no-PKG-INFO tree still builds (never the
# 'fatal: not a git repository' string).
BASE_VERSION = _git_base_version() or "0.0.0_dev"


def get_version_with_cuda_torch():
    """Generate version string with CUDA and PyTorch version suffix.

    Example: 0.0.0_dev+cu128.torch2.10
    """
    # sdist rebuild: reuse the already-resolved PKG-INFO version verbatim.
    pkg_version = _read_pkg_info_version()
    if pkg_version:
        return pkg_version
    try:
        import torch

        # Get CUDA version (e.g., "12.8" -> "128")
        cuda_version = torch.version.cuda
        if cuda_version:
            cuda_version = cuda_version.replace(".", "")
        else:
            cuda_version = "cpu"

        # Get PyTorch version (e.g., "2.10.0" -> "2.10")
        torch_version = torch.__version__.split("+")[0]  # Remove any existing suffix
        torch_major_minor = ".".join(torch_version.split(".")[:2])

        return f"{BASE_VERSION}+cu{cuda_version}.torch{torch_major_minor}"
    except ImportError:
        # torch not installed, return base version
        return BASE_VERSION


TOOLS_VERSION = get_version_with_cuda_torch()


def get_requirements(filename):
    """Load dependency packages from specified requirements file"""
    with open(filename) as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith(("#", "-"))
        ]


setup(
    name="angelslim",
    version=TOOLS_VERSION,
    description=("A toolkit for compress llm model."),
    long_description="Tools for llm model compression",
    url="https://github.com/Tencent/AngelSlim",
    author="Tencent Author",
    # Core dependencies: installed by default
    install_requires=get_requirements("requirements/requirements.txt"),
    # Define optional dependency groups
    extras_require={
        # Install all optional features: pip install angelslim[all]
        # NOTE: [all] intentionally does NOT include [sparse]. The sparse extra
        # raises the transformers floor to >=5.8 (Qwen3.5 gated-attention modeling),
        # higher than the main pin; folding it into [all] would force that floor on
        # every [all] user. So that sparse's higher floor stays opt-in, sparse
        # is installed separately: pip install angelslim[sparse].
        "all": (
            get_requirements("requirements/requirements_speculative.txt")
            + get_requirements("requirements/requirements_diffusion.txt")
            + get_requirements("requirements/requirements_multimodal.txt")
            + get_requirements("requirements/requirements_benchmark.txt")
            + get_requirements("requirements/requirements_mcore_qad.txt")
        ),
        # Install speculative sampling functionality: pip install angelslim[speculative]
        "speculative": get_requirements("requirements/requirements_speculative.txt"),
        # Install Diffusion functionality: pip install angelslim[diffusion]
        "diffusion": get_requirements("requirements/requirements_diffusion.txt"),
        # Install multimodal functionality: pip install angelslim[multimodal]
        "multimodal": get_requirements("requirements/requirements_multimodal.txt"),
        # Install benchmark functionality: pip install angelslim[benchmark]
        "benchmark": get_requirements("requirements/requirements_benchmark.txt"),
        # Install the Megatron-Core scale-only QAT/QAD backend
        "mcore-qad": get_requirements("requirements/requirements_mcore_qad.txt"),
        # Install sparse-attention functionality: pip install angelslim[sparse]
        "sparse": get_requirements("requirements/requirements_sparse.txt"),
    },
    packages=find_packages(),
    # The vendored MInference CUDA index extension is JIT-compiled from its
    # C++/CUDA sources at runtime, so those non-.py files MUST ship in the wheel
    # (find_packages only collects .py). Without this, `pip install .[sparse]`
    # then a `minference` run fails because csrc/*.cpp/*.cu are absent. Also ship
    # the kernel NOTICE files (MIT attribution travels with the vendored code).
    package_data={
        "angelslim.compressor.sparsity.algorithms.minference.kernels": [
            "csrc/*.cpp",
            "csrc/*.cu",
            "csrc/*.h",
            "NOTICE",
        ],
        # FlexPrefill is pure-Triton (no csrc), but its NOTICE carries the MIT
        # attribution for the vendored kernel and must travel with the wheel.
        "angelslim.compressor.sparsity.algorithms.flexprefill.kernels": [
            "NOTICE",
        ],
        # XAttention vendors two .py kernel files + a NOTICE (MIT attribution);
        # the .py travel via find_packages, the NOTICE must be listed here.
        "angelslim.compressor.sparsity.algorithms.xattention.kernels": [
            "NOTICE",
        ],
        # FlashPrefill is CLEAN-ROOM (no vendored upstream code), so its NOTICE
        # lives at the package root (no kernels/ subdir) and records the
        # clean-room provenance + paper citation. It must travel with the wheel.
        "angelslim.compressor.sparsity.algorithms.flashprefill": [
            "NOTICE",
        ],
        # VecAttention: original Apache-2.0 code; its NOTICE (package root) records
        # the external vllm-flash-attention fork dependency (flash-attn + CUTLASS
        # licenses) and the .pkl autotune-cache provenance. The Triton kernel loads
        # pre-tuned launch configs from the pickled caches at runtime (falls back to
        # safe defaults if absent, but ship them so the tuned path is the default).
        "angelslim.compressor.sparsity.algorithms.vecattention": [
            "NOTICE",
        ],
        "angelslim.compressor.sparsity.algorithms.vecattention.ops": [
            "cache/*.pkl",
        ],
    },
    include_package_data=True,
    python_requires=">=3.10",
    # PyPI package information.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="License for AngelSlim",
    keywords=("Tencent large language model model-optimize compression toolkit."),
)
