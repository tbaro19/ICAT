"""
PaliGemma Model Wrapper for Vision-Language Tasks
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


class PaliGemmaWrapper:
    """Wrapper for PaliGemma model"""
    
    def __init__(self, model_name: str = "google/paligemma-3b-pt-224", device: str = "cuda"):
        """
        Initialize PaliGemma model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        print(f"Loading PaliGemma model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load with bfloat16 for efficiency
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        print(f"✓ PaliGemma loaded on {device}")
    
    def generate_caption(self, image, max_length: int = 77):
        """
        Generate caption for an image using PaliGemma
        
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
        
        # Use simple but effective prompt - PaliGemma works best with short prompts
        prompt = "<image>caption en"
        
        try:
            # Prepare inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            # Generate caption with more tokens
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,  # Increase for longer captions
                    do_sample=False,
                    repetition_penalty=1.2,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode output and remove prompt
            full_output = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Remove the prompt prefix if present
            if full_output.startswith("<image>caption en"):
                full_output = full_output[len("<image>caption en"):].strip()  # Remove "<image>caption en" prefix
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return full_output.strip()
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during caption generation, clearing cache and retrying...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Retry with simpler generation
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50, num_beams=1)
            full_output = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Remove the prompt prefix if present
            if full_output.startswith("describe"):
                full_output = full_output[8:].strip()  # Remove "describe" + space
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return full_output.strip()
    
    def compute_similarity(self, images, texts):
        """
        Compute similarity between images and texts
        
        Args:
            images: Batch of images [B, C, H, W] or single PIL Image
            texts: List of text strings or single string
            
        Returns:
            Similarity scores as 1D tensor [B]
        """
        from PIL import Image as PILImage
        
        # Handle single image
        if isinstance(images, PILImage.Image):
            images = [images]
            single_input = True
        else:
            single_input = False
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Convert tensor images to PIL
        pil_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                if img.ndim == 4:
                    img = img.squeeze(0)
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(PILImage.fromarray(img_np))
            else:
                pil_images.append(img)
        
        try:
            # Compute similarity scores
            scores = []
            for pil_img, text in zip(pil_images, texts):
                # Use caption generation as proxy for similarity
                # Add image token prefix
                text_with_token = f"<image>{text}"
                inputs = self.processor(text=text_with_token, images=pil_img, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use negative loss as similarity proxy
                    logits = outputs.logits
                    # Simple heuristic: higher confidence in prediction = higher similarity
                    score = torch.sigmoid(logits.mean()).item()
                
                scores.append(score)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = torch.tensor(scores, dtype=torch.float32)
            return result[0] if single_input else result
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during similarity computation")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Return dummy scores
            result = torch.ones(len(images), dtype=torch.float32) * 0.5
            return result[0] if single_input else result
    
    def get_text_embeddings(self, texts):
        """
        Get text embeddings
        
        Args:
            texts: List of text strings or single string
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # PaliGemma requires both text and images, so provide a dummy image
        from PIL import Image as PILImage
        dummy_image = PILImage.new('RGB', (224, 224), color=(128, 128, 128))
        dummy_images = [dummy_image] * len(texts)
        
        # Add image token prefix to all texts
        texts_with_tokens = [f"<image>{text}" for text in texts]
        
        # Process with dummy images
        inputs = self.processor(text=texts_with_tokens, images=dummy_images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            # Get embeddings from language model
            embeddings = self.model.language_model.get_input_embeddings()(inputs.input_ids)
        
        return embeddings
    
    def get_image_embeddings(self, images):
        """
        Get image embeddings
        
        Args:
            images: Batch of images [B, C, H, W] or single PIL Image
            
        Returns:
            Image embeddings tensor
        """
        from PIL import Image as PILImage
        
        # Convert to PIL if needed
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                batch_size = images.shape[0]
            else:
                images = images.unsqueeze(0)
                batch_size = 1
            
            pil_images = []
            for i in range(batch_size):
                img = images[i]
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(PILImage.fromarray(img_np))
        elif isinstance(images, PILImage.Image):
            pil_images = [images]
        else:
            pil_images = images
        
        # PaliGemma requires both text and images, so provide dummy text with image token
        dummy_texts = ["<image>"] * len(pil_images)
        
        # Process images with dummy text
        inputs = self.processor(text=dummy_texts, images=pil_images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get vision embeddings
            pixel_values = inputs.pixel_values
            embeddings = self.model.vision_tower(pixel_values)
        
        return embeddings
