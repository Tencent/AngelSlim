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

from enum import Enum
from typing import Dict


class ChatTemplateType(Enum):
    """Supported chat template types."""

    QWEN3 = "qwen3"


# String to ChatTemplateType mapping
CHAT_TEMPLATE_TYPE_MAPPING = {
    "qwen3": ChatTemplateType.QWEN3,
}


class ChatTemplate:
    """Chat template configuration for a specific model type."""

    def __init__(self, user_header: str, assistant_header: str):
        self.user_header = user_header
        self.assistant_header = assistant_header

    def to_dict(self) -> Dict[str, str]:
        """Convert template to dictionary format."""
        return {
            "user_header": self.user_header,
            "assistant_header": self.assistant_header,
        }


class ChatTemplateManager:
    """Manager for chat templates of different model types."""

    def __init__(self):
        self._templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[ChatTemplateType, ChatTemplate]:
        """Initialize predefined chat templates."""
        return {
            ChatTemplateType.QWEN3: ChatTemplate(
                user_header="<|im_start|>user\n",
                assistant_header="<|im_start|>assistant\n",
            )
        }

    def get_template(self, chat_template_type: ChatTemplateType) -> ChatTemplate:
        """
        Get chat template for specified chat template type.

        Args:
            chat_template_type: The chat template type to get template for

        Returns:
            ChatTemplate instance

        Raises:
            ValueError: If chat template type is not supported
        """
        if chat_template_type not in self._templates:
            raise ValueError(f"Unsupported chat template type: {chat_template_type}")

        return self._templates[chat_template_type]

    def get_template_dict(self, chat_template_type: ChatTemplateType) -> Dict[str, str]:
        """
        Get chat template as dictionary for specified chat template type.

        Args:
            chat_template_type: The chat template type to get template for

        Returns:
            Dictionary containing template configuration
        """
        template = self.get_template(chat_template_type)
        return template.to_dict()

    def list_supported_types(self) -> list[str]:
        """
        List all supported chat template types.

        Returns:
            List of supported chat template type names
        """
        return [template_type.value for template_type in self._templates.keys()]


# Global template manager instance
template_manager = ChatTemplateManager()


# Convenience functions for backward compatibility
def get_template(chat_template_type: ChatTemplateType) -> Dict[str, str]:
    """Get chat template dictionary for specified chat template type."""
    return template_manager.get_template_dict(chat_template_type)


def list_supported_chat_template_types() -> list[str]:
    """List all supported chat template types."""
    return template_manager.list_supported_types()


def string_to_chat_template_type(template_type_str: str) -> ChatTemplateType:
    """
    Convert string to ChatTemplateType enum.

    Args:
        template_type_str: String representation of chat template type

    Returns:
        ChatTemplateType enum

    Raises:
        ValueError: If chat template type string is not supported
    """
    if template_type_str not in CHAT_TEMPLATE_TYPE_MAPPING:
        supported_types = list(CHAT_TEMPLATE_TYPE_MAPPING.keys())
        raise ValueError(
            f"Unsupported chat template type: {template_type_str}. "
            f"Supported types: {supported_types}"
        )

    return CHAT_TEMPLATE_TYPE_MAPPING[template_type_str]


def get_supported_chat_template_type_strings() -> list[str]:
    """
    Get list of supported chat template type strings for command line arguments.

    Returns:
        List of supported chat template type strings
    """
    return list(CHAT_TEMPLATE_TYPE_MAPPING.keys())
