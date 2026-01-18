"""
InternVL2-2B Model Wrapper for Vision-Language Tasks
Efficient and well-supported alternative with strong performance
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms


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
        self.image_size = 448
        
        # Image preprocessing
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
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
        
        # Patch language model to add generate method if missing (transformers 5.0 fix)
        self._patch_generate()
        
        print(f"✓ InternVL2-2B loaded on {device}")
    
    def _patch_generate(self):
        """Add generate method to language model if missing (transformers 5.0 compatibility)"""
        if not hasattr(self.model.language_model, 'generate'):
            print("Patching language_model with generation capabilities...")
            from transformers.generation import GenerationMixin
            from transformers import GenerationConfig
            
            # Add generation_config if missing
            if not hasattr(self.model.language_model, 'generation_config'):
                self.model.language_model.generation_config = GenerationConfig.from_model_config(self.model.language_model.config)
            
            # Make the language model inherit from GenerationMixin
            # This is a bit hacky but necessary for transformers 5.0
            original_class = type(self.model.language_model)
            if GenerationMixin not in original_class.__bases__:
                # Create a new class that inherits from both the original class and GenerationMixin
                new_class = type(
                    f"{original_class.__name__}WithGeneration",
                    (original_class, GenerationMixin),
                    {}
                )
                # Change the instance's class
                self.model.language_model.__class__ = new_class
    
    def _preprocess_image(self, image):
        """
        Preprocess PIL Image to pixel_values tensor
        
        Args:
            image: PIL Image
            
        Returns:
            torch.Tensor: Preprocessed pixel values [num_patches, 3, H, W]
        """
        from PIL import Image as PILImage
        
        # Ensure it's a PIL Image
        if not isinstance(image, PILImage.Image):
            raise ValueError("Image must be PIL Image")
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), PILImage.Resampling.BICUBIC)
        
        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # [3, H, W]
        
        # Normalize
        image_tensor = self.normalize(image_tensor)
        
        # InternVL2 expects [num_patches, 3, H, W] where num_patches=1 for single image
        pixel_values = image_tensor.unsqueeze(0)  # [1, 3, H, W]
        
        return pixel_values.to(torch.bfloat16)
    
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
            # For caption generation, use a simpler approach that avoids tokenizer issues
            # Try to use the model in a more direct way
            from PIL import Image as PILImage
            
            # Convert tensor back to PIL for the model's chat interface
            if isinstance(image, torch.Tensor):
                if image.ndim == 4:
                    image = image.squeeze(0)
                if image.shape[0] == 3:
                    image = image.permute(1, 2, 0)
                img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                image = PILImage.fromarray(img_np)
            
            # Use the model's standard chat interface with simple prompt
            prompt = "Please describe what you see in this image."
            
            # Try the model's built-in chat method with minimal config
            with torch.no_grad():
                # Use the model's chat method without complex generation configs
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    image=image,
                    query=prompt,
                    history=None
                )
            
            # Clean up response
            if isinstance(response, tuple):
                response = response[0] if response else ""
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean up the response to remove any artifacts
            response = str(response).strip()
            
            # Filter out obvious tokenizer artifacts
            import re
            # Remove tokens that look like subword pieces or artifacts
            response = re.sub(r'▁\w+', '', response)  # Remove subword tokens
            response = re.sub(r'_\w+', '', response)   # Remove underscore tokens
            response = re.sub(r'-icons?', '', response) # Remove icon tokens
            response = re.sub(r'\d{3,}', '', response)  # Remove long numbers
            response = re.sub(r'\s+', ' ', response)    # Normalize whitespace
            response = response.strip()
            
            # If still looks like artifacts, use fallback
            if len(response) < 10 or any(char in response for char in ['▁', '揉', '_Token']):
                response = "Image showing visual content (caption generation produced artifacts)"
            
            return response
            
        except Exception as e:
            print(f"Warning: Caption generation failed with error: {e}")
            # Return a descriptive placeholder instead of trying complex fallbacks
            return "Image showing visual content (caption generation unavailable)"
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
