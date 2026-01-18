"""
DeepSeek-VL2-Tiny Model Wrapper for Vision-Language Tasks
Efficient 1B parameter model with strong performance

Note: Requires transformers with DeepSeek-VL2 support:
  pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git
  or use the model's custom loading code
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


class DeepSeekVL2Wrapper:
    """Wrapper for DeepSeek-VL2-Tiny model"""
    
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
        print("Note: This model requires transformers with deepseek_vl_v2 support")
        
        try:
            # Load model with trust_remote_code
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model.eval()
            
            print(f"✓ DeepSeek-VL2-Tiny loaded on {device}")
            
        except ValueError as e:
            if "deepseek_vl_v2" in str(e):
                print(f"\n⚠️ Error: DeepSeek-VL2 model type not recognized by transformers")
                print(f"This model requires special installation:")
                print(f"  Option 1: pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git")
                print(f"  Option 2: pip install git+https://github.com/huggingface/transformers.git")
                print(f"\nPlease install the required package and try again.")
                raise RuntimeError(f"DeepSeek-VL2 requires updated transformers library") from e
            else:
                raise
    
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
            # DeepSeek-VL2 uses the generate method with images
            with torch.no_grad():
                response, _ = self.model.generate(
                    image=image,
                    text="Describe this image in detail.",
                    tokenizer=self.tokenizer,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=3
                )
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response.strip()
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during caption generation, clearing cache and retrying...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Retry with simpler generation
            with torch.no_grad():
                response, _ = self.model.generate(
                    image=image,
                    text="Describe this image.",
                    tokenizer=self.tokenizer,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1
                )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response.strip()
    
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
            # Get model outputs with logits
            with torch.no_grad():
                # Use the model's forward pass to get logits
                outputs = self.model.forward_with_image(
                    image=image,
                    text="Describe this image.",
                    tokenizer=self.tokenizer,
                    return_logits=True
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
