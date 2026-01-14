"""
Perturbation generation and application utilities
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Optional


class PerturbationGenerator:
    """Generate and apply adversarial perturbations"""
    
    def __init__(self, image_shape: Tuple[int, int, int], epsilon: float = 0.1, target_size: Optional[int] = None):
        """
        Initialize perturbation generator
        
        Args:
            image_shape: Shape of images (C, H, W) - low resolution for search
            epsilon: Maximum perturbation magnitude (L-inf norm)
            target_size: Target size to upsample perturbations to (if None, use image_shape size)
        """
        self.image_shape = image_shape
        self.epsilon = epsilon
        self.solution_dim = np.prod(image_shape)
        self.target_size = target_size
        
    def solutions_to_perturbations(self, solutions: np.ndarray) -> np.ndarray:
        """
        Convert flat solution vectors to image perturbations
        
        Args:
            solutions: Flat solution vectors [batch_size, solution_dim]
            
        Returns:
            Image perturbations [batch_size, C, H, W] - upsampled to target_size if specified
        """
        batch_size = solutions.shape[0]
        perturbations = solutions.reshape(batch_size, *self.image_shape)
        
        # Upsample if target_size specified
        if self.target_size is not None and self.image_shape[1] != self.target_size:
            perturbations_torch = torch.from_numpy(perturbations).float()
            perturbations_torch = F.interpolate(
                perturbations_torch,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            )
            perturbations = perturbations_torch.numpy()
        
        # Clip to epsilon ball
        perturbations = np.clip(perturbations, -self.epsilon, self.epsilon)
        
        return perturbations
    
    def apply_perturbation(self, images: Union[torch.Tensor, np.ndarray], 
                          perturbations: np.ndarray) -> np.ndarray:
        """
        Apply perturbations to images
        
        Args:
            images: Original images [batch_size, C, H, W] or [C, H, W]
            perturbations: Perturbations to apply [batch_size, C, H, W]
            
        Returns:
            Perturbed images clipped to valid range [0, 1]
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        # Handle single image
        if images.ndim == 3:
            images = images[np.newaxis, :]
        
        # Broadcast if needed
        if images.shape[0] == 1 and perturbations.shape[0] > 1:
            images = np.repeat(images, perturbations.shape[0], axis=0)
        
        # Apply perturbation
        perturbed = images + perturbations
        
        # Clip to valid image range [0, 1]
        perturbed = np.clip(perturbed, 0.0, 1.0)
        
        return perturbed
    
    def random_perturbation(self, batch_size: int = 1) -> np.ndarray:
        """
        Generate random perturbations
        
        Args:
            batch_size: Number of perturbations to generate
            
        Returns:
            Random perturbations [batch_size, solution_dim]
        """
        perturbations = np.random.uniform(
            -self.epsilon, 
            self.epsilon, 
            size=(batch_size, self.solution_dim)
        )
        return perturbations
    
    def project_to_epsilon_ball(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Project perturbations to L-inf epsilon ball
        
        Args:
            perturbations: Perturbations [batch_size, ...]
            
        Returns:
            Projected perturbations
        """
        return np.clip(perturbations, -self.epsilon, self.epsilon)
    
    def compute_linf_norm(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute L-inf norm of perturbations
        
        Args:
            perturbations: Perturbations [batch_size, solution_dim]
            
        Returns:
            L-inf norms [batch_size]
        """
        if perturbations.ndim == 2:
            return np.max(np.abs(perturbations), axis=1)
        else:
            # Reshape first
            batch_size = perturbations.shape[0]
            flat = perturbations.reshape(batch_size, -1)
            return np.max(np.abs(flat), axis=1)
    
    def compute_l2_norm(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute L2 norm of perturbations
        
        Args:
            perturbations: Perturbations [batch_size, solution_dim]
            
        Returns:
            L2 norms [batch_size]
        """
        if perturbations.ndim == 2:
            return np.linalg.norm(perturbations, axis=1)
        else:
            batch_size = perturbations.shape[0]
            flat = perturbations.reshape(batch_size, -1)
            return np.linalg.norm(flat, axis=1)
