"""
InternVL2-2B Model Wrapper for Vision-Language Tasks
Efficient and well-supported alternative with strong performance
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class InternVL2Wrapper:
    """Wrapper for InternVL2-2B model"""
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL2-2B", device: str = "cuda"):
        """
        Initialize InternVL2-2B model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        print(f"Loading InternVL2-2B model: {model_name}")
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        print(f"✓ InternVL2-2B loaded on {device}")
    
    def generate_caption(self, image, max_length: int = 100):
        """
        Generate caption for an image using InternVL2
        
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
            # InternVL2 uses simple prompt format
            prompt = "<image>\nDescribe this image in detail."
            
            # Generate caption
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=None,
                    image=image,
                    msgs=[{'role': 'user', 'content': prompt}],
                    generation_config=dict(
                        max_new_tokens=128,
                        do_sample=False,
                        num_beams=3
                    )
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
            prompt = "<image>\nDescribe this image."
            
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=None,
                    image=image,
                    msgs=[{'role': 'user', 'content': prompt}],
                    generation_config=dict(
                        max_new_tokens=64,
                        do_sample=False,
                        num_beams=1
                    )
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
            # Build prompt
            prompt = "<image>\nDescribe this image."
            
            # Get model outputs with logits
            with torch.no_grad():
                # InternVL2 chat returns text, we need direct model forward pass
                # This is a simplified version - may need adjustment for actual logit extraction
                pixel_values = self.model.extract_feature(image).to(self.device)
                
                # Tokenize prompt
                input_ids = self.tokenizer(
                    prompt, 
                    return_tensors='pt'
                ).input_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
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
            vocab_size = self.tokenizer.vocab_size
            return torch.zeros(vocab_size, device=self.device)
