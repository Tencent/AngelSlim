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

from abc import ABC, abstractmethod

__all__ = ["TransformBase"]


class TransformBase(ABC):
    """Abstract base class for model weight transforms (e.g. SpinQuant).

    Subclasses must implement `run()`. The lifecycle is:
        1. TransformFactory.create(quant_model, quant_config)  -> TransformBase
        2. transform.run()      - apply transform (PTQ: fuse into weights)
        3. transform.convert()  - fuse hooks into weights after QAT training (optional)
        4. transform.save()     - save transformed model (optional)
    """

    def __init__(self, quant_model, quant_config):
        self.quant_model = quant_model
        self.config = quant_config

    @abstractmethod
    def run(self):
        """Apply the transform to the model weights."""

    def convert(self, **kwargs):
        """Fuse online rotation hooks into weights after QAT training.

        Override in subclasses that support QAT mode.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement convert()")

    def save(self, save_path: str = None):
        """Save the transformed model.

        Override in subclasses to implement actual saving logic.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")
