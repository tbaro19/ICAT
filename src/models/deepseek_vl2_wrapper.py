"""
DeepSeek-VL2-Tiny Model Wrapper for Vision-Language Tasks
Efficient 1B parameter model with strong performance

Installation:
  git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
  cd DeepSeek-VL2
  pip install -e .
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM

# Monkey patch torch.library.register_fake to skip torchvision operator registration
# This fixes: RuntimeError: operator torchvision::nms does not exist
_original_register_fake = torch.library.register_fake

def _patched_register_fake(schema_or_name, *args, **kwargs):
    """Skip torchvision operator registration that causes errors with torch 2.5.1"""
    if isinstance(schema_or_name, str) and 'torchvision::' in schema_or_name:
        # Return dummy decorator for torchvision operators
        def dummy_decorator(fn):
            return fn
        return dummy_decorator
    return _original_register_fake(schema_or_name, *args, **kwargs)

torch.library.register_fake = _patched_register_fake

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class DeepSeekVL2Wrapper:
    """Wrapper for DeepSeek-VL2-Tiny model using official API"""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl2-tiny", device: str = "cuda"):
        """
        Initialize DeepSeek-VL2-Tiny model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        print(f"Loading DeepSeek-VL2-Tiny model: {model_name}")
        print("Using official DeepSeek-VL2 API")
        
        # Load processor (handles tokenization and image processing)
        self.processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        
        # Load model using the correct class
        # Use float16 instead of bfloat16 for better compatibility with older GPUs (T4, etc.)
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16  # Changed from bfloat16 for T4 compatibility
        )
        
        if device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        
        self.model.eval()
        
        print(f"✓ DeepSeek-VL2-Tiny loaded on {device}")
    
    def generate_caption(self, image, max_length: int = 100):
        """
        Generate caption for an image using DeepSeek-VL2
        
        Args:
            image: PIL Image or torch.Tensor [C, H, W] or [B, C, H, W]
            max_length: Maximum caption length
            
        Returns:
            Generated caption string
        """
        from PIL import Image as PILImage
        
        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            # Remove batch dimension if present
            if image.ndim == 4:
                image = image.squeeze(0)  # [B, C, H, W] -> [C, H, W]
            
            if image.shape[0] == 3:  # [C, H, W]
                image = image.permute(1, 2, 0)  # [H, W, C]
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            image = PILImage.fromarray(img_np)
        
        try:
            # Prepare conversation format
            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>\nDescribe this image in detail.",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # Prepare inputs
            with torch.no_grad():
                prepare_inputs = self.processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True,
                    system_prompt=""
                ).to(self.model.device)
                
                # Get image embeddings
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                
                # Generate response
                outputs = self.model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_length,
                    do_sample=False,
                    use_cache=True
                )
                
                # Decode response
                answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
                # Extract answer content (remove prompt)
                if "<|Assistant|>" in answer:
                    answer = answer.split("<|Assistant|>")[-1].strip()
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return answer
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during caption generation, clearing cache and retrying...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Retry with simpler generation
            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>\nDescribe this image.",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            with torch.no_grad():
                prepare_inputs = self.processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True,
                    system_prompt=""
                ).to(self.model.device)
                
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                
                outputs = self.model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=64,
                    do_sample=False
                )
                
                answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                if "<|Assistant|>" in answer:
                    answer = answer.split("<|Assistant|>")[-1].strip()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return answer
    
    def generate_captions(self, images):
        """
        Generate captions for a batch of images
        
        Args:
            images: List of PIL Images or torch.Tensor [B, C, H, W]
            
        Returns:
            List of generated captions
        """
        # Convert tensor batch to list of PIL images
        if isinstance(images, torch.Tensor):
            from PIL import Image as PILImage
            image_list = []
            for i in range(images.shape[0]):
                img = images[i]  # [C, H, W]
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0)  # [H, W, C]
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                image_list.append(PILImage.fromarray(img_np))
            images = image_list
        
        # Process each image individually for stability
        captions = []
        for image in images:
            caption = self.generate_caption(image)
            captions.append(caption)
        
        return captions
    
    def extract_logits(self, image, target_text: str = None):
        """
        Extract next-token logits for jailbreak fitness computation
        
        Args:
            image: PIL Image or torch.Tensor
            target_text: Target caption (optional, unused)
            
        Returns:
            Logits tensor [vocab_size]
        """
        from PIL import Image as PILImage
        
        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            image = PILImage.fromarray(img_np)
        
        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>\nDescribe this image.",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            with torch.no_grad():
                # Prepare inputs
                prepare_inputs = self.processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True,
                    system_prompt=""
                ).to(self.model.device)
                
                # Get image embeddings
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                
                # Forward pass to get logits
                outputs = self.model.language(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    return_dict=True
                )
                
                # Get logits at the last position
                logits = outputs.logits[0, -1, :]  # [vocab_size]
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return logits
            
        except Exception as e:
            print(f"Warning: Error during logit extraction: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Return zeros if extraction fails
            vocab_size = len(self.tokenizer)
            return torch.zeros(vocab_size, device=self.device)
