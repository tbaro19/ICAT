"""
Base class for Vision-Language Models
"""
from abc import ABC, abstractmethod
import torch
from typing import List, Union
import numpy as np


class VLMBase(ABC):
    """Abstract base class for Vision-Language Models"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and processor"""
        pass
    
    @abstractmethod
    def generate_captions(self, images: Union[torch.Tensor, np.ndarray]) -> List[str]:
        """
        Generate captions for a batch of images
        
        Args:
            images: Batch of images as tensor or numpy array
            
        Returns:
            List of generated captions
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, images: Union[torch.Tensor, np.ndarray], 
                          texts: List[str]) -> torch.Tensor:
        """
        Compute similarity scores between images and texts
        
        Args:
            images: Batch of images
            texts: List of text captions
            
        Returns:
            Similarity scores tensor
        """
        pass
    
    def preprocess_images(self, images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for model input
        
        Args:
            images: Batch of images
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        if images.device != self.device:
            images = images.to(self.device)
            
        return images
    
    @torch.no_grad()
    def encode_images(self, images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Encode images to embedding space
        
        Args:
            images: Batch of images
            
        Returns:
            Image embeddings
        """
        images = self.preprocess_images(images)
        return self.model.encode_image(images)
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to embedding space
        
        Args:
            texts: List of text captions
            
        Returns:
            Text embeddings
        """
        return self.model.encode_text(texts)
