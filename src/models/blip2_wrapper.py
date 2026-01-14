"""
BLIP-2 Vision-Language Model Wrapper
"""
import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import List, Union
import numpy as np


class BLIP2Wrapper(nn.Module):
    """Wrapper for BLIP-2 model"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
        """
        Initialize BLIP-2 model
        
        Args:
            model_name: BLIP-2 model variant
            device: Device to run model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        print(f"Loading BLIP-2 model: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()
        
        print(f"✓ BLIP-2 loaded on {device}")
    
    def generate_caption(self, images, max_length: int = 50):
        """
        Generate captions for images
        
        Args:
            images: Either a single PIL Image, or a batch of images [B, C, H, W] in [0, 1]
            max_length: Maximum caption length
            
        Returns:
            Single caption string (if input is PIL Image) or List of generated captions (if batch)
        """
        from PIL import Image as PILImage
        
        # Handle single PIL Image
        if isinstance(images, PILImage.Image):
            try:
                # BLIP-2 uses unconditional generation for captioning (no text prompt)
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        min_length=40,
                        num_beams=5,
                        length_penalty=1.5,
                        repetition_penalty=1.2,
                        do_sample=False,
                        early_stopping=False,
                    )
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return caption.strip()
            except torch.cuda.OutOfMemoryError:
                print("Warning: OOM during caption generation, retrying with beam=1...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_length, num_beams=1)
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return caption.strip()
        
        # Handle batch of torch.Tensors
        # Handle single tensor with batch dimension
        if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] == 1:
            images = images.squeeze(0)  # [1, C, H, W] -> [C, H, W]
            # Convert to PIL and process as single image
            if images.shape[0] == 3:  # [C, H, W]
                images = images.permute(1, 2, 0)  # [H, W, C]
            img_np = (images.cpu().numpy() * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(img_np)
            # BLIP-2 uses unconditional generation (no text prompt)
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    min_length=40,
                    num_beams=5,
                    length_penalty=1.5,
                    repetition_penalty=1.2,
                    early_stopping=False
                )
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return caption.strip()
        
        pil_images = []
        for img in images:
            if img.shape[0] == 3:  # [C, H, W]
                img = img.permute(1, 2, 0)  # [H, W, C]
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(PILImage.fromarray(img_np))
        
        # BLIP-2 uses unconditional generation for captioning (no text prompts)
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        
        # Generate captions with parameters for longer, detailed output
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                min_length=40,
                num_beams=5,
                length_penalty=1.5,
                repetition_penalty=1.2,
                do_sample=False,
                early_stopping=False
            )
            captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return captions
    
    def compute_similarity(self, images, texts) -> torch.Tensor:
        """
        Compute similarity between images and texts
        
        Args:
            images: torch.Tensor [B, C, H, W] or single PIL Image
            texts: List of text captions or single string
            
        Returns:
            Similarity scores - tensor [B] for batch, float for single
        """
        from PIL import Image as PILImage
        
        # Handle single image/text case
        if isinstance(images, PILImage.Image):
            if isinstance(texts, str):
                return self._compute_single_similarity(images, texts)
            else:
                raise ValueError("Single image requires single text string")
        
        # Handle batched case
        if isinstance(images, torch.Tensor):
            batch_size = images.shape[0]
            similarities = []
            
            for i in range(batch_size):
                img_tensor = images[i]  # [C, H, W]
                text = texts[i] if isinstance(texts, list) else texts
                
                # Convert tensor to PIL
                if img_tensor.shape[0] == 3:  # [C, H, W]
                    img_tensor = img_tensor.permute(1, 2, 0)  # [H, W, C]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_image = PILImage.fromarray(img_np)
                
                # Compute similarity
                sim = self._compute_single_similarity(pil_image, text)
                similarities.append(sim)
            
            return torch.tensor(similarities, device=self.device)
        
        raise ValueError(f"Unsupported image type: {type(images)}")
    
    def _compute_single_similarity(self, image, text: str) -> float:
        """Helper method to compute similarity for a single image-text pair"""
        try:
            # Use caption generation to check similarity
            prompt = f'Does this image show: "{text}"? Answer yes or no.'
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10)
                response = self.processor.decode(outputs[0], skip_special_tokens=True).strip().lower()
                
                # Map response to similarity score
                if "yes" in response:
                    return 0.85  # High similarity
                elif "no" in response:
                    return 0.25  # Low similarity
                else:
                    return 0.5  # Uncertain
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during similarity computation")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.5
    
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get text embeddings using language model's text encoder
        
        Args:
            texts: List of text strings
            
        Returns:
            Text embeddings [len(texts), embed_dim]
        """
        embeddings = []
        for text in texts:
            try:
                text_inputs = self.processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                with torch.no_grad():
                    # Use the language model to get text embeddings
                    text_outputs = self.model.language_model.get_input_embeddings()(text_inputs.input_ids)
                    # Mean pooling over sequence length
                    text_features = text_outputs.mean(dim=1)
                    embeddings.append(text_features)
                
                # Clean up cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print("Warning: OOM during text embedding")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                embeddings.append(torch.zeros(1, 2048).to(self.device))
        
        return torch.cat(embeddings, dim=0)
    
    def get_image_embeddings(self, images: List) -> torch.Tensor:
        """
        Get image embeddings using vision model
        
        Args:
            images: List of PIL Images or torch.Tensors
            
        Returns:
            Image embeddings [len(images), embed_dim]
        """
        from PIL import Image as PILImage
        
        embeddings = []
        for image in images:
            try:
                # Convert to PIL if needed
                if isinstance(image, torch.Tensor):
                    if image.shape[0] == 3:
                        image = image.permute(1, 2, 0)
                    img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                    image = PILImage.fromarray(img_np)
                
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    # Get vision features
                    vision_outputs = self.model.vision_model(
                        pixel_values=image_inputs.pixel_values,
                        return_dict=True,
                    )
                    image_features = vision_outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(image_features)
                
                # Clean up cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print("Warning: OOM during image embedding")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                embeddings.append(torch.zeros(1, 1408).to(self.device))
        
        return torch.cat(embeddings, dim=0)
    
    def forward(self, images: torch.Tensor) -> List[str]:
        """Forward pass generates captions"""
        return self.generate_caption(images)
