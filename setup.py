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
import subprocess

from setuptools import find_packages, setup

TOOLS_VERSION = None

if "main" in subprocess.getoutput("git branch"):
    TOOLS_VERSION = "0.0.0_dev"
else:
    tag_list = subprocess.getoutput("git tag").split("\n")
    TOOLS_VERSION = tag_list[-1]


def get_requirements():
    """from requirements.txt load dependency package"""
    with open("requirements.txt") as f:
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
    install_requires=get_requirements(),
    packages=find_packages(),
    python_requires=">=3.0",
    # PyPI package information.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
