"""
Vision-Language Model Wrappers for Black-box Attacks
"""
from .vlm_base import VLMBase
from .blip2_wrapper import BLIP2Wrapper
from .deepseek_vl2_wrapper import DeepSeekVL2Wrapper
from .qwen2vl_wrapper import Qwen2VLWrapper

__all__ = ['VLMBase', 'BLIP2Wrapper', 'DeepSeekVL2Wrapper', 'Qwen2VLWrapper']
