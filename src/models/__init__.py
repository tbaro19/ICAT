"""
Vision-Language Model Wrappers for Black-box Attacks
"""
from .vlm_base import VLMBase
from .blip2_wrapper import BLIP2Wrapper
from .paligemma_wrapper import PaliGemmaWrapper
from .moondream2_wrapper import Moondream2Wrapper

__all__ = ['VLMBase', 'BLIP2Wrapper', 'PaliGemmaWrapper', 'Moondream2Wrapper']
