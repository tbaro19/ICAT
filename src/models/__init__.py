"""
Vision-Language Model Wrappers for Black-box Attacks
"""
from .vlm_base import VLMBase
from .tinyllava_wrapper import TinyLLaVAWrapper

# Qwen2VL requires transformers >= 4.41.0
try:
    from .qwen2vl_wrapper import Qwen2VLWrapper
    __all__ = ['TinyLLaVAWrapper', 'Qwen2VLWrapper']
except ImportError:
    __all__ = ['TinyLLaVAWrapper']
    print("Warning: Qwen2VL not available with current transformers version")
