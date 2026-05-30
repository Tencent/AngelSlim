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


class PluginManager:
    _plugin_registry = {}

    def __init__(self):
        self.plugins = {}

    @classmethod
    def plugin(cls, name):
        def decorator(plugin_class):
            cls.register_plugin_class(name, plugin_class)
            return plugin_class

        return decorator

    @classmethod
    def register_plugin_class(cls, name: str, plugin_class):
        cls._plugin_registry[name] = plugin_class

    def register_plugin(self, name: str, **kwargs):
        if name not in self._plugin_registry:
            raise ValueError(
                f"Unknown plugin type: {name}. " f"Available: {list(self._plugin_registry.keys())}"
            )

        plugin_class = self._plugin_registry[name]
        plugin = plugin_class(**kwargs)

        self.plugins[name] = plugin

    def call_before_train(self, **kwargs):
        for p in self.plugins.values():
            p.before_train(**kwargs)

    def call_after_train(self, **kwargs):
        for p in self.plugins.values():
            p.after_train(**kwargs)
