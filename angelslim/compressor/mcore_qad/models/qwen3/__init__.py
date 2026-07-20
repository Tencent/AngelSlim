from angelslim.compressor.mcore_qad.models.base import register
from angelslim.compressor.mcore_qad.models.qwen3.config import qwen3_moe_config
from angelslim.compressor.mcore_qad.models.qwen3.convert import qwen3_to_mcore

register("qwen3_moe", qwen3_moe_config, qwen3_to_mcore)
