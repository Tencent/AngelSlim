"""AngelSlim's distributed QAT/QAD backend for Megatron-Core.

Design pillars (see README):
  * Fake-quant only (BF16 compute), weights frozen, only quantizer *scales* are trainable.
  * Quant strategy = compose(Format x ScaleScheme x ScaleSource).
  * Parallel-correct scales (TP/EP/CP aware) derived automatically.
  * Megatron-Core production backend; per-model adapters under models/<name>/.
  * Loss = lm_weight*LM + distill_weight*distill (QAD = quant-off teacher / quant-on student).
"""

__version__ = "0.0.1.dev0"
