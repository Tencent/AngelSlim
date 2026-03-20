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


class BasePlugin:
    def __init__(self, config=None, quant_model=None):
        self.config = config
        self.quant_model = quant_model

    def before_train(self, **kwargs):
        """Execute before training starts"""
        pass

    def after_train(self, **kwargs):
        """Execute after training ends"""
        pass
