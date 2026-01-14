"""
Fitness functions for measuring attack effectiveness
"""
import torch
import numpy as np
from typing import List, Union
from bert_score import score as bert_score


class FitnessFunction:
    """Calculate fitness (attack effectiveness) for perturbations"""
    
    def __init__(self, metric: str = 'clip_similarity', device='cuda', chunk_size: int = 4):
        """
        Initialize fitness function
        
        Args:
            metric: Type of metric ('clip_similarity', 'bertscore', 'negative_similarity')
            device: Device for computation
            chunk_size: Size of chunks for batch processing to avoid OOM
        """
        self.metric = metric
        self.device = device
        self.chunk_size = chunk_size
        
    def compute_fitness(self, 
                       original_captions: List[str],
                       perturbed_images: np.ndarray,
                       vlm_model,
                       original_embeddings: torch.Tensor = None) -> np.ndarray:
        """
        Compute fitness values for perturbed images
        
        Higher fitness = More effective attack (more semantic destruction)
        
        Args:
            original_captions: Original ground-truth captions
            perturbed_images: Perturbed images [batch_size, C, H, W]
            vlm_model: VLM model instance
            original_embeddings: Cached original text embeddings (optional)
            
        Returns:
            Fitness values [batch_size]
        """
        if self.metric == 'clip_similarity':
            return self._clip_similarity_fitness(
                original_captions, 
                perturbed_images, 
                vlm_model,
                original_embeddings
            )
        elif self.metric == 'bertscore':
            return self._bertscore_fitness(
                original_captions,
                perturbed_images,
                vlm_model
            )
        elif self.metric == 'negative_similarity':
            return self._negative_similarity_fitness(
                original_captions,
                perturbed_images,
                vlm_model,
                original_embeddings
            )
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _clip_similarity_fitness(self,
                                original_captions: List[str],
                                perturbed_images: np.ndarray,
                                vlm_model,
                                original_embeddings: torch.Tensor = None) -> np.ndarray:
        """
        Fitness based on CLIP similarity between perturbed images and original captions
        Lower similarity = Higher fitness (successful attack)
        Processes in chunks to avoid OOM
        
        Args:
            original_captions: Original captions
            perturbed_images: Perturbed images
            vlm_model: VLM model
            original_embeddings: Cached text embeddings
            
        Returns:
            Fitness values (1 - similarity)
        """
        # Convert to tensor
        if isinstance(perturbed_images, np.ndarray):
            perturbed_images = torch.from_numpy(perturbed_images).float()
        
        batch_size = perturbed_images.shape[0]
        all_fitness = []
        
        # Process in chunks to avoid OOM
        for i in range(0, batch_size, self.chunk_size):
            end_idx = min(i + self.chunk_size, batch_size)
            chunk_images = perturbed_images[i:end_idx]
            chunk_captions = original_captions[i:end_idx]
            
            # Compute similarity for chunk
            with torch.no_grad():
                similarity = vlm_model.compute_similarity(
                    chunk_images, 
                    chunk_captions
                )
                
                # compute_similarity now returns 1D tensor [B] directly
                # Each element is the similarity between image[i] and caption[i]
                if isinstance(similarity, torch.Tensor):
                    scores = similarity.cpu().numpy()
                elif isinstance(similarity, (int, float)):
                    # Single value case
                    scores = np.array([similarity])
                else:
                    # Already numpy
                    scores = np.array(similarity)
            
            # Fitness = 1 - similarity (higher is better for attack)
            chunk_fitness = 1.0 - scores
            all_fitness.append(chunk_fitness)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        fitness = np.concatenate(all_fitness) if len(all_fitness) > 1 else all_fitness[0]
        
        return fitness
    
    def _negative_similarity_fitness(self,
                                    original_captions: List[str],
                                    perturbed_images: np.ndarray,
                                    vlm_model,
                                    original_embeddings: torch.Tensor = None) -> np.ndarray:
        """
        Fitness as negative similarity (to be maximized)
        
        Args:
            original_captions: Original captions
            perturbed_images: Perturbed images
            vlm_model: VLM model
            original_embeddings: Cached text embeddings
            
        Returns:
            Negative similarity values
        """
        if isinstance(perturbed_images, np.ndarray):
            perturbed_images = torch.from_numpy(perturbed_images).float()
        
        with torch.no_grad():
            similarity = vlm_model.compute_similarity(
                perturbed_images,
                original_captions
            )
            
            batch_size = similarity.shape[0]
            if batch_size == 1:
                scores = similarity.squeeze()
            else:
                scores = torch.diagonal(similarity)
            
            scores = scores.cpu().numpy()
        
        # Return negative similarity
        return -scores
    
    def _bertscore_fitness(self,
                          original_captions: List[str],
                          perturbed_images: np.ndarray,
                          vlm_model) -> np.ndarray:
        """
        Fitness based on BERTScore between original and perturbed captions
        
        Note: This requires caption generation, which may not be available
        for all VLMs (CLIP/SigLIP are contrastive, not generative)
        
        Args:
            original_captions: Original captions
            perturbed_images: Perturbed images
            vlm_model: VLM model
            
        Returns:
            Fitness values (1 - BERTScore)
        """
        # This would require a generative model
        # For CLIP/SigLIP, fall back to similarity-based fitness
        print("Warning: BERTScore requires generative captions. Using CLIP similarity instead.")
        return self._clip_similarity_fitness(
            original_captions,
            perturbed_images,
            vlm_model
        )


class VietnameseFitnessFunction(FitnessFunction):
    """Fitness function specialized for Vietnamese text"""
    
    def __init__(self, metric: str = 'clip_similarity', device='cuda'):
        """
        Initialize Vietnamese fitness function
        
        Args:
            metric: Type of metric
            device: Device for computation
        """
        super().__init__(metric=metric, device=device)
        
        # Load Vietnamese BERT for semantic similarity if needed
        if metric == 'vibert_similarity':
            from transformers import AutoModel, AutoTokenizer
            self.vibert_model = AutoModel.from_pretrained("vinai/phobert-base")
            self.vibert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.vibert_model.to(device)
            self.vibert_model.eval()
