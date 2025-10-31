import argparse

import torch
from diffusers import AutoModel, DiffusionPipeline

from angelslim.compressor.diffusion import DynamicDiTQuantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diffusion Model FP8 inference & quantization harness"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help=(
            "Model name from the HuggingFace Hub or local path for loading "
            "a Diffusion Pipeline (default: black-forest-labs/FLUX.1-schnell)"
        ),
    )
    parser.add_argument(
        "--fp8-model-load-path",
        type=str,
        default=None,
        help=(
            "Path to a custom transformer (e.g., quantized weights directory). "
            "If set, always uses this transformer."
        ),
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="fp8-per-tensor",
        choices=[
            "fp8-per-tensor",
            "fp8-per-block",
            "fp8-per-token",
            "fp8-per-tensor-weight-only",
        ],
        help="Select FP8 quantization type",
    )
    parser.add_argument(
        "--include-patterns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Quantize only layers whose names match these patterns "
            "(supports substring or regex)"
        ),
    )
    parser.add_argument(
        "--exclude-patterns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Exclude layers whose names match these patterns "
            "(supports substring or regex)"
        ),
    )
    parser.add_argument(
        "--fp8-model-save-path",
        type=str,
        default=None,
        help="If set, exports the quantized model and fp8_scales.safetensors",
    )
    parser.add_argument(
        "--prompt", type=str, default="A cat holding a sign that says hello world"
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> DiffusionPipeline:
    """
    Build and return a DiffusionPipeline with optional quantized transformer support.

    Args:
        args: Argument namespace from argparse

    Returns:
        DiffusionPipeline object
    """
    transformer_path = args.fp8_model_load_path
    model_name = args.model_name_or_path

    # Load pipeline with quantized transformer if provided
    if transformer_path is not None:
        dit = DynamicDiTQuantizer.load_quantized_model(AutoModel, transformer_path)
        pipe = DiffusionPipeline.from_pretrained(
            model_name, transformer=dit, torch_dtype=torch.bfloat16
        )
    else:
        # Load pipeline from scratch
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    return pipe


def main():
    args = parse_args()

    # Initialize quantizer
    quantizer = DynamicDiTQuantizer(
        quant_type=args.quant_type,
        include_patterns=args.include_patterns,
        exclude_patterns=args.exclude_patterns,
    )

    # Build pipeline (with or without pre-quantized transformer)
    pipe = build_pipeline(args)

    quantizer.convert_linear(pipe.transformer, scale=args.fp8_model_load_path)

    # Export quantized model if save path is specified
    if args.fp8_model_save_path is not None:
        quantizer.export_quantized_weight(
            pipe.transformer, save_path=args.fp8_model_save_path
        )
        return

    # Run inference
    device = args.device
    pipe.to(device)

    generator = torch.Generator(device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        max_sequence_length=256,
        generator=generator,
    ).images[0]

    # Generate output filename based on model name and parameters
    model_name_clean = (
        args.model_name_or_path.replace("/", "-")
        if args.model_name_or_path
        else "diffusion"
    )
    quant_type_clean = args.quant_type.replace("/", "-")
    out_name = (
        f"{model_name_clean}_{quant_type_clean}_" f"{args.height}x{args.width}.png"
    )
    image.save(out_name)


if __name__ == "__main__":
    main()
