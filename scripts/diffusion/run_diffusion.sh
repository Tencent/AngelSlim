# Example: Diffusion FP8 quantization/inference
# Tip: Copy the following commands to your terminal; modify paths/parameters as needed.

# Quantize the model online and run inference
# quant-type options: fp8-per-tensor, fp8-per-block, fp8-per-token, fp8-per-tensor-weight-only
python scripts/diffusion/run_diffusion.py \
  --model-name-or-path /path_or_name/to/model \
  --quant-type fp8-per-tensor \
  --prompt "A cat holding a sign that says hello world" \
  --height 1024 --width 1024 --steps 4 --guidance 0.0 --seed 0

# # Quantize the model and export the quantized weight
# python scripts/diffusion/run_diffusion.py \
#   --model-name-or-path /path_or_name/to/model \
#   --fp8-model-save-path /path/to/save/quantized_model \
#   --quant-type fp8-per-tensor \

# # Load an already quantized model, then run inference
# python scripts/diffusion/run_diffusion.py \
#   --model-name-or-path /path_or_name/to/model \
#   --fp8-model-load-path /path/to/quantized_model \
#   --quant-type fp8-per-tensor \
#   --prompt "A cat holding a sign that says hello world" \
#   --height 1024 --width 1024 --steps 4 --guidance 0.0 --seed 0
