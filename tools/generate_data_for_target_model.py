import argparse

from angelslim.compressor.speculative.train.data import data_generation_work_flow


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate conversational data using OpenAI API"
    )
    parser.add_argument(
        "--data_name_or_path",
        type=str,
        required=True,
        help="Path to input data file (ShareGPT format)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output data directory"
    )
    parser.add_argument(
        "--num_threads", type=int, default=256, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Maximum tokens per response"
    )
    parser.add_argument(
        "--base_port", type=int, default=6000, help="Base port for API servers"
    )
    parser.add_argument(
        "--max_clients", type=int, default=8, help="Maximum number of API clients"
    )
    parser.add_argument(
        "--data_shard_size", type=int, default=50000, help="Size of each data shard"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="sharegpt",
        help="Data format (sharegpt or ultrachat)",
        choices=["sharegpt", "ultrachat"],
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    data_generation_work_flow(args)


if __name__ == "__main__":
    main()
