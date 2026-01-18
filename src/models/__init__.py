"""
Vision-Language Model Wrappers for Black-box Attacks
"""
from .vlm_base import VLMBase
from .tinyllava_wrapper import TinyLLaVAWrapper
from .qwen2vl_wrapper import Qwen2VLWrapper

__all__ = ['TinyLLaVAWrapper', 'Qwen2VLWrapper']
