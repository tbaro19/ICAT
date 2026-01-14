"""
Adaptive Sigma Scheduler for Quality-Diversity Algorithms
Handles stagnation detection and dynamic parameter adjustment
"""
import numpy as np
from typing import List, Dict, Optional
from ribs.emitters import EmitterBase


class AdaptiveSigmaScheduler:
    """
    Manages adaptive sigma and epsilon adjustments to overcome stagnation.
    
    Detects when the archive stops growing and temporarily increases exploration
    by boosting sigma (step size) and epsilon (perturbation bound).
    """
    
    def __init__(
        self,
        emitters: List[EmitterBase],
        perturbation_generator,
        baseline_sigma: float = 0.02,
        baseline_epsilon: float = 0.12,
        stagnation_threshold: int = 15,
        sigma_multiplier: float = 2.0,
        epsilon_boost: float = 0.10
    ):
        """
        Initialize Adaptive Sigma Scheduler
        
        Args:
            emitters: List of pyribs emitters to manage
            perturbation_generator: PerturbationGenerator instance to adjust epsilon
            baseline_sigma: Initial/baseline sigma value (default: 0.02)
            baseline_epsilon: Initial/baseline epsilon value (default: 0.12)
            stagnation_threshold: Number of iterations without progress before intervention (default: 15)
            sigma_multiplier: Factor to multiply sigma by when stagnating (default: 2.0)
            epsilon_boost: Percentage to boost epsilon by when stagnating (default: 0.10 = 10%)
        """
        self.emitters = emitters
        self.pert_gen = perturbation_generator
        
        # Baseline values
        self.baseline_sigma = baseline_sigma
        self.baseline_epsilon = baseline_epsilon
        
        # Adaptive parameters
        self.stagnation_threshold = stagnation_threshold
        self.sigma_multiplier = sigma_multiplier
        self.epsilon_boost = epsilon_boost
        
        # State tracking
        self.current_sigma = baseline_sigma
        self.current_epsilon = baseline_epsilon
        self.last_num_elites = 0
        self.stagnation_counter = 0
        self.is_boosted = False
        
        # History for logging
        self.history = {
            'iteration': [],
            'num_elites': [],
            'sigma': [],
            'epsilon': [],
            'stagnation_counter': [],
            'is_boosted': []
        }
        
    def update(self, iteration: int, num_elites: int) -> Dict[str, any]:
        """
        Update scheduler state based on current archive status.
        
        Args:
            iteration: Current iteration number
            num_elites: Current number of elites in archive
            
        Returns:
            Dictionary with current status and any changes made
        """
        status = {
            'changed': False,
            'action': 'none',
            'sigma': self.current_sigma,
            'epsilon': self.current_epsilon,
            'stagnation_counter': self.stagnation_counter
        }
        
        # Check for progress
        if num_elites > self.last_num_elites:
            # Progress detected!
            if self.is_boosted:
                # Reset to baseline values
                self._reset_to_baseline()
                status['changed'] = True
                status['action'] = 'reset'
                print(f"\n  🔄 [Adaptive Sigma] Progress detected! Resetting to baseline:")
                print(f"     Sigma: {self.current_sigma:.4f} | Epsilon: {self.current_epsilon:.4f}")
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            
        else:
            # No progress - increment stagnation counter
            self.stagnation_counter += 1
            
            # Check if we need to boost
            if self.stagnation_counter >= self.stagnation_threshold and not self.is_boosted:
                self._apply_boost()
                status['changed'] = True
                status['action'] = 'boost'
                print(f"\n  🚀 [Adaptive Sigma] Stagnation detected after {self.stagnation_counter} iterations!")
                print(f"     Boosting: Sigma {self.baseline_sigma:.4f} → {self.current_sigma:.4f}")
                print(f"              Epsilon {self.baseline_epsilon:.4f} → {self.current_epsilon:.4f}")
            
            # Continue boosting if already in boosted mode and still stagnating
            elif self.stagnation_counter >= self.stagnation_threshold and self.is_boosted:
                # Further boost sigma (keep doubling)
                if self.stagnation_counter % self.stagnation_threshold == 0:
                    self._apply_additional_boost()
                    status['changed'] = True
                    status['action'] = 'additional_boost'
                    print(f"\n  🚀🚀 [Adaptive Sigma] Continued stagnation - additional boost!")
                    print(f"       Sigma: {self.current_sigma:.4f} | Epsilon: {self.current_epsilon:.4f}")
        
        # Update last count
        self.last_num_elites = num_elites
        
        # Log history
        self.history['iteration'].append(iteration)
        self.history['num_elites'].append(num_elites)
        self.history['sigma'].append(self.current_sigma)
        self.history['epsilon'].append(self.current_epsilon)
        self.history['stagnation_counter'].append(self.stagnation_counter)
        self.history['is_boosted'].append(self.is_boosted)
        
        status['sigma'] = self.current_sigma
        status['epsilon'] = self.current_epsilon
        status['stagnation_counter'] = self.stagnation_counter
        
        return status
    
    def _apply_boost(self):
        """Apply initial boost to sigma and epsilon"""
        self.current_sigma = self.baseline_sigma * self.sigma_multiplier
        self.current_epsilon = self.baseline_epsilon * (1.0 + self.epsilon_boost)
        
        # Update emitters
        self._update_emitter_sigma(self.current_sigma)
        
        # Update perturbation generator
        self.pert_gen.epsilon = self.current_epsilon
        
        self.is_boosted = True
    
    def _apply_additional_boost(self):
        """Apply additional boost when already in boosted mode"""
        # Double sigma again
        self.current_sigma *= self.sigma_multiplier
        
        # Don't increase epsilon further - cap it
        # (epsilon shouldn't exceed certain physical bounds)
        
        # Update emitters
        self._update_emitter_sigma(self.current_sigma)
    
    def _reset_to_baseline(self):
        """Reset sigma and epsilon to baseline values"""
        self.current_sigma = self.baseline_sigma
        self.current_epsilon = self.baseline_epsilon
        
        # Update emitters
        self._update_emitter_sigma(self.current_sigma)
        
        # Update perturbation generator
        self.pert_gen.epsilon = self.current_epsilon
        
        self.is_boosted = False
        self.stagnation_counter = 0
    
    def _update_emitter_sigma(self, new_sigma: float):
        """Update sigma for all emitters"""
        for emitter in self.emitters:
            # Different emitter types have different sigma attributes
            if hasattr(emitter, '_sigma'):
                # GaussianEmitter uses _sigma
                emitter._sigma = new_sigma
            elif hasattr(emitter, 'sigma'):
                # Some emitters might use sigma directly
                emitter.sigma = new_sigma
            elif hasattr(emitter, '_sigma0'):
                # EvolutionStrategyEmitter uses _sigma0
                if hasattr(emitter, '_opt'):
                    # Update the optimizer's sigma
                    emitter._opt.sigma0 = new_sigma
            elif hasattr(emitter, 'iso_sigma'):
                # IsoLineEmitter
                emitter.iso_sigma = new_sigma
            
            # For ES emitters with optimizers, try to update them
            if hasattr(emitter, '_opt'):
                if hasattr(emitter._opt, 'sigma0'):
                    emitter._opt.sigma0 = new_sigma
    
    def get_history(self) -> Dict[str, List]:
        """Get history of adaptive adjustments"""
        return self.history
    
    def get_current_state(self) -> Dict[str, any]:
        """Get current scheduler state"""
        return {
            'sigma': self.current_sigma,
            'epsilon': self.current_epsilon,
            'stagnation_counter': self.stagnation_counter,
            'is_boosted': self.is_boosted,
            'last_num_elites': self.last_num_elites
        }
    
    def save_history(self, save_path: str):
        """Save history to file"""
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self.history, f)
        print(f"  ✓ Saved adaptive sigma history to {save_path}")
