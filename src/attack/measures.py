"""
Measure functions (Behavior Characteristics) for QD diversity
"""
import numpy as np
from scipy.fft import fft2, fftshift
from typing import Tuple


class MeasureFunction:
    """Compute behavior characteristics for MAP-Elites grid"""
    
    def __init__(self, 
                 image_shape: Tuple[int, int, int],
                 measure_types: Tuple[str, str] = ('linf_norm', 'spectral_energy')):
        """
        Initialize measure function
        
        Args:
            image_shape: Shape of images (C, H, W)
            measure_types: Types of measures to compute
        """
        self.image_shape = image_shape
        self.measure_types = measure_types
        
    def compute_measures(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute all measures for perturbations
        
        Args:
            perturbations: Perturbations [batch_size, C, H, W] or [batch_size, solution_dim]
            
        Returns:
            Measures [batch_size, n_measures]
        """
        # Reshape if needed
        if perturbations.ndim == 2:
            batch_size = perturbations.shape[0]
            perturbations = perturbations.reshape(batch_size, *self.image_shape)
        
        measures = []
        
        for measure_type in self.measure_types:
            if measure_type == 'linf_norm':
                m = self._compute_linf_norm(perturbations)
            elif measure_type == 'l2_norm':
                m = self._compute_l2_norm(perturbations)
            elif measure_type == 'spectral_energy':
                m = self._compute_spectral_energy(perturbations)
            elif measure_type == 'spatial_sparsity':
                m = self._compute_spatial_sparsity(perturbations)
            elif measure_type == 'spatial_centroid_x':
                m = self._compute_spatial_centroid(perturbations, axis=0)
            elif measure_type == 'spatial_centroid_y':
                m = self._compute_spatial_centroid(perturbations, axis=1)
            elif measure_type == 'frequency_ratio':
                m = self._compute_frequency_ratio(perturbations)
            else:
                raise ValueError(f"Unknown measure type: {measure_type}")
            
            measures.append(m)
        
        # Stack measures
        measures = np.stack(measures, axis=1)
        return measures
    
    def _compute_linf_norm(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute L-infinity norm (max absolute value)
        
        Args:
            perturbations: [batch_size, C, H, W]
            
        Returns:
            L-inf norms [batch_size]
        """
        batch_size = perturbations.shape[0]
        flat = perturbations.reshape(batch_size, -1)
        return np.max(np.abs(flat), axis=1)
    
    def _compute_l2_norm(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute L2 norm
        
        Args:
            perturbations: [batch_size, C, H, W]
            
        Returns:
            L2 norms [batch_size]
        """
        batch_size = perturbations.shape[0]
        flat = perturbations.reshape(batch_size, -1)
        return np.linalg.norm(flat, axis=1)
    
    def _compute_spectral_energy(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute spectral energy (high-frequency content via FFT)
        
        Args:
            perturbations: [batch_size, C, H, W]
            
        Returns:
            Spectral energy values [batch_size] in range [0, 1]
        """
        batch_size = perturbations.shape[0]
        energies = []
        
        for i in range(batch_size):
            # Average over channels
            pert = perturbations[i]  # [C, H, W]
            
            # Compute FFT for each channel
            channel_energies = []
            for c in range(pert.shape[0]):
                fft = fft2(pert[c])
                fft_shift = fftshift(fft)
                magnitude = np.abs(fft_shift)
                
                # Compute high-frequency energy (outer region)
                h, w = magnitude.shape
                center_h, center_w = h // 2, w // 2
                radius = min(h, w) // 4
                
                # Create mask for high frequencies
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_w)**2 + (y - center_h)**2) > radius**2
                
                # High-frequency energy ratio
                high_freq_energy = np.sum(magnitude[mask])
                total_energy = np.sum(magnitude)
                
                if total_energy > 0:
                    ratio = high_freq_energy / total_energy
                else:
                    ratio = 0.0
                
                channel_energies.append(ratio)
            
            # Average over channels
            avg_energy = np.mean(channel_energies)
            energies.append(avg_energy)
        
        return np.array(energies)
    
    def _compute_spatial_sparsity(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute spatial sparsity (how localized the perturbation is)
        
        Args:
            perturbations: [batch_size, C, H, W]
            
        Returns:
            Sparsity values [batch_size] in range [0, 1]
        """
        batch_size = perturbations.shape[0]
        sparsities = []
        
        for i in range(batch_size):
            pert = perturbations[i]
            
            # Compute magnitude across channels
            magnitude = np.sqrt(np.sum(pert**2, axis=0))  # [H, W]
            
            # Threshold to find significant perturbations
            threshold = 0.01 * np.max(magnitude)
            significant = magnitude > threshold
            
            # Sparsity = fraction of pixels with significant perturbation
            sparsity = np.mean(significant)
            sparsities.append(sparsity)
        
        return np.array(sparsities)
    
    def _compute_spatial_centroid(self, perturbations: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute spatial centroid of perturbation
        
        Args:
            perturbations: [batch_size, C, H, W]
            axis: 0 for x-axis, 1 for y-axis
            
        Returns:
            Centroid values [batch_size] normalized to [0, 1]
        """
        batch_size = perturbations.shape[0]
        centroids = []
        
        for i in range(batch_size):
            pert = perturbations[i]
            
            # Compute magnitude
            magnitude = np.sqrt(np.sum(pert**2, axis=0))  # [H, W]
            
            # Compute centroid
            if axis == 0:  # x-axis (width)
                coords = np.arange(magnitude.shape[1])
                weights = np.sum(magnitude, axis=0)
            else:  # y-axis (height)
                coords = np.arange(magnitude.shape[0])
                weights = np.sum(magnitude, axis=1)
            
            total_weight = np.sum(weights)
            if total_weight > 0:
                centroid = np.sum(coords * weights) / total_weight
                # Normalize to [0, 1]
                centroid = centroid / len(coords)
            else:
                centroid = 0.5  # Center if no perturbation
            
            centroids.append(centroid)
        
        return np.array(centroids)
    
    def _compute_frequency_ratio(self, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute ratio of high to low frequency content
        
        Args:
            perturbations: [batch_size, C, H, W]
            
        Returns:
            Frequency ratios [batch_size]
        """
        # Similar to spectral energy but returns raw ratio
        return self._compute_spectral_energy(perturbations)
