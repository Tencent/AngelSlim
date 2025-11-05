import concurrent.futures
import json
import logging
import os
import random
from typing import Dict, List, Optional

import tqdm
from datasets import load_dataset

from angelslim.utils.lazy_imports import openai

from .data_utils import convert_sharegpt_data, convert_ultrachat_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OpenAIClientPool:
    """Manages a pool of OpenAI clients for load balancing."""

    def __init__(self, base_port: int = 6000, max_clients: int = 10):
        """
        Initialize the client pool.

        Args:
            base_port: Starting port number for API servers
            max_clients: Maximum number of clients to create
        """
        self.clients = []
        self._initialize_clients(base_port, max_clients)

    def _initialize_clients(self, base_port: int, max_clients: int) -> None:
        """Initialize available OpenAI clients."""
        for i in range(max_clients):
            base_url = f"http://localhost:{base_port + i}/v1"
            client = openai.OpenAI(base_url=base_url, api_key="EMPTY")

            try:
                model_id = client.models.list().data[0].id
                logger.info(f"Connected to {base_url} with model {model_id}")
                self.clients.append(client)
            except Exception as e:
                logger.warning(f"Failed to connect to {base_url}: {e}")
                break

        if not self.clients:
            raise RuntimeError("No available OpenAI clients found")

        logger.info(f"Initialized {len(self.clients)} clients")

    def get_client(self, idx: int):
        """Get a client using round-robin load balancing."""
        return self.clients[idx % len(self.clients)]

    def __len__(self) -> int:
        return len(self.clients)


class DataGenerator:
    """Generates conversational data using OpenAI API."""

    def __init__(self, client_pool: OpenAIClientPool, max_tokens: int = 2048):
        """
        Initialize the data generator.

        Args:
            client_pool: Pool of OpenAI clients
            max_tokens: Maximum tokens per response
        """
        self.client_pool = client_pool
        self.temperature = get_random_temperature()
        self.max_tokens = max_tokens

    def _convert_messages(
        self, messages: List[Dict]
    ) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Convert ShareGPT format messages to OpenAI format.

        Args:
            messages: Messages in ShareGPT format

        Returns:
            Tuple of (converted_messages, output_messages, remaining_messages)
        """
        converted_messages = []

        # Handle system message
        if messages and messages[0].get("role") == "system":
            system_msg = {"role": "system", "content": messages[0]["content"]}
            converted_messages.append(system_msg)
            messages = messages[1:]

        # Skip if first message is not from human
        if messages and messages[0].get("role") != "user":
            messages = messages[1:]

        return converted_messages, messages

    def _generate_response(
        self, client, messages: List[Dict], **kwargs
    ) -> Optional[str]:
        """
        Generate a response using the OpenAI API.

        Args:
            client: OpenAI client instance
            messages: Conversation messages

        Returns:
            Generated response or None if failed
        """
        try:
            enable_thinking = kwargs.get("enable_thinking", False)

            model_name = client.models.list().data[0].id
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                },
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return None

    def generate_conversation(
        self,
        messages: List[Dict],
        id: str,
        idx: int,
        **kwargs,
    ) -> Optional[List[Dict]]:
        """
        Generate a complete conversation.

        Args:
            messages: Input messages in ShareGPT format
            idx: Index for load balancing

        Returns:
            Generated conversation or None if failed
        """
        try:
            client = self.client_pool.get_client(idx)
            converted_messages, remaining_messages = self._convert_messages(messages)

            # Process user messages (every other message starting from index 0)
            for message in remaining_messages[::2]:
                assert message.get("role") in ["user", "assistant", "system"]
                if message.get("role") != "user":
                    continue

                # Add user message
                user_msg = {"role": "user", "content": message["content"]}
                converted_messages.append(user_msg)

                # Generate assistant response
                response = self._generate_response(client, converted_messages, **kwargs)
                if response is None:
                    break

                assistant_msg = {"role": "assistant", "content": response}
                converted_messages.append(assistant_msg)

            # Validate output
            if len(converted_messages) <= 1:
                return None

            new_row = {"id": id, "conversations": converted_messages}
            return new_row

        except Exception as e:
            logger.error(f"Failed to generate conversation: {e}")
            return None


def save_conversation(output_path: str, conversation: List[Dict]) -> None:
    """
    Save a conversation to file.

    Args:
        output_path: Output file path
        conversation: Conversation to save
    """
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")


def count_existing_lines(file_path: str) -> int:
    """
    Count existing lines in output file.

    Args:
        file_path: Path to output file

    Returns:
        Number of existing lines
    """
    if not os.path.exists(file_path):
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return len(f.readlines())
    except Exception as e:
        logger.error(f"Failed to count existing lines: {e}")
        return 0


def get_random_temperature() -> float:
    choices = [0.0, 0.3, 0.5, 0.7, 1.0]
    weights = [4, 1, 1, 1, 3]
    return random.choices(choices, weights=weights)[0]


def data_generation_work_flow(args):
    """Main execution function."""
    # args = parse_arguments()

    # Load input data
    logger.info(f"Loading data from {args.data_name_or_path}")
    try:
        dataset = load_dataset(args.data_name_or_path, split="all")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Convert data format
    if args.data_format == "sharegpt":
        convert_func = convert_sharegpt_data
    elif args.data_format == "ultrachat":
        convert_func = convert_ultrachat_data
    else:
        raise ValueError(f"Invalid data format: {args.data_format}")
    dataset = dataset.map(convert_func, desc="Converting data format..")

    # Initialize client pool
    try:
        client_pool = OpenAIClientPool(
            base_port=args.base_port, max_clients=args.max_clients
        )
    except Exception as e:
        logger.error(f"Failed to initialize client pool: {e}")
        return

    # Initialize data generator
    generator = DataGenerator(client_pool=client_pool, max_tokens=args.max_tokens)

    os.makedirs(args.output_dir, exist_ok=True)
    for start in range(0, len(dataset), args.data_shard_size):
        end = min(start + args.data_shard_size, len(dataset))
        output_path = os.path.join(args.output_dir, f"data_{start}-{end}.jsonl")
        current_dataset = dataset.select(range(start, end))

        # Process data with thread pool
        # Check for existing progress
        start_idx = count_existing_lines(output_path)
        if start_idx > 0:
            logger.info(f"Resuming from index {start_idx}")
        logger.info(
            f"Processing {len(current_dataset) - start_idx} "
            f"samples with {args.num_threads} threads"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = []

            for idx, row in enumerate(
                current_dataset.select(range(start_idx, len(current_dataset)))
            ):
                future = executor.submit(
                    generator.generate_conversation,
                    row["conversations"],
                    row["id"],
                    idx,
                )
                futures.append(future)

            # Process results with progress bar
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Generating conversations",
            ):
                try:
                    new_row = future.result()
                    if new_row:
                        save_conversation(output_path, new_row)
                except Exception as e:
                    logger.error(f"Future execution failed: {e}")

    logger.info("Data generation completed")
