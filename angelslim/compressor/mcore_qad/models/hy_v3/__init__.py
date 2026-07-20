from angelslim.compressor.mcore_qad.models.base import register
from angelslim.compressor.mcore_qad.models.hy_v3.config import hy_v3_config
from angelslim.compressor.mcore_qad.models.hy_v3.convert import hy_v3_to_mcore

register("hy_v3", hy_v3_config, hy_v3_to_mcore)
