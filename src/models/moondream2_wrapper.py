"""
Moondream2 Model Wrapper for Vision-Language Tasks
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


class Moondream2Wrapper:
    """Wrapper for Moondream2 vision-language model (1.6B parameters, efficient for T4 GPU)"""
    
    def __init__(self, model_name: str = "vikhyatk/moondream2", device: str = "cuda"):
        """
        Initialize Moondream2 model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        print(f"Loading Moondream2 model: {model_name}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        self.model.eval()
        
        print(f"✓ Moondream2 loaded on {device}")
    
    def generate_caption(self, image, max_length: int = 77):
        """
        Generate caption for an image using Moondream2
        
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
        
        # Use Moondream2's query method for caption generation
        try:
            # Encode image
            enc_image = self.model.encode_image(image)
            
            # Generate caption with detailed prompt
            prompt = "Provide a detailed description of this image, including all visible objects, people, actions, and scene context."
            
            with torch.no_grad():
                caption = self.model.answer_question(
                    enc_image, 
                    prompt, 
                    self.tokenizer,
                    max_new_tokens=100
                )
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return caption.strip()
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during Moondream2 caption generation, returning fallback...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "A scene with various objects and elements"
        except Exception as e:
            print(f"Warning: Error during Moondream2 caption generation: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "A scene with various objects and elements"
    
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
                # Encode image
                enc_image = self.model.encode_image(pil_img)
                
                # Ask verification question
                question = f'Does this image show: "{text}"? Answer only yes or no.'
                
                with torch.no_grad():
                    response = self.model.answer_question(
                        enc_image,
                        question,
                        self.tokenizer,
                        max_new_tokens=5
                    ).strip().lower()
                
                # Map yes/no to similarity score
                if 'yes' in response:
                    score = 0.9
                elif 'no' in response:
                    score = 0.2
                else:
                    score = 0.5
                
                scores.append(score)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = torch.tensor(scores, dtype=torch.float32)
            return result[0] if single_input else result
            
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during Moondream2 similarity computation")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = torch.ones(len(images), dtype=torch.float32) * 0.5
            return result[0] if single_input else result
        except Exception as e:
            print(f"Warning: Error during Moondream2 similarity computation: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        
        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                # Get embeddings from model's text embeddings
                embeddings = self.model.text_model.get_input_embeddings()(inputs.input_ids)
            
            return embeddings
            
        except Exception as e:
            print(f"Warning: Error getting text embeddings: {e}")
            # Return dummy embeddings
            return torch.zeros(len(texts), 512, 768).to(self.device)
    
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
        
        try:
            # Get image embeddings
            embeddings = []
            for pil_img in pil_images:
                with torch.no_grad():
                    enc_image = self.model.encode_image(pil_img)
                embeddings.append(enc_image)
            
            # Stack embeddings
            if len(embeddings) == 1:
                return embeddings[0]
            else:
                return torch.stack(embeddings)
                
        except Exception as e:
            print(f"Warning: Error getting image embeddings: {e}")
            # Return dummy embeddings
            return torch.zeros(len(pil_images), 729, 1152).to(self.device)
