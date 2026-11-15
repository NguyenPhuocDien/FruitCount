"""DEIMv2 inference modules only; upstream arithmetic is unchanged."""
from .core import GLOBAL_CONFIG, create
from .backbone.dinov3_adapter import DINOv3STAs
from .deim.deim import DEIM
from .deim.hybrid_encoder import HybridEncoder
from .deim.deim_decoder import DEIMTransformer
from .deim.postprocessor import PostProcessor
