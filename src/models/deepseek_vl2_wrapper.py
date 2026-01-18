"""
DeepSeek-VL2-Tiny Model Wrapper for Vision-Language Tasks
Lightweight alternative to PaliGemma with strong performance
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


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
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model with bfloat16 for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
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
            # DeepSeek-VL2 uses conversational format
            conversation = [
                {
                    "role": "User",
                    "content": "<image>\nDescribe this image in detail.",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]
            
            # Prepare inputs
            inputs = self.processor(
                conversation,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=3,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
            
            # Decode output
            output_text = self.processor.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            if "Assistant:" in output_text:
                output_text = output_text.split("Assistant:")[-1].strip()
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output_text.strip()
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during caption generation, clearing cache and retrying...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Retry with simpler generation (beam_size=1)
            conversation = [
                {
                    "role": "User",
                    "content": "<image>\nDescribe this image.",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]
            
            inputs = self.processor(
                conversation,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1
                )
            
            output_text = self.processor.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            if "Assistant:" in output_text:
                output_text = output_text.split("Assistant:")[-1].strip()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output_text.strip()
    
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
            # Build conversation
            conversation = [
                {
                    "role": "User",
                    "content": "<image>\nDescribe this image.",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]
            
            # Prepare inputs
            inputs = self.processor(
                conversation,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
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
            return torch.zeros(self.model.config.vocab_size, device=self.device)
