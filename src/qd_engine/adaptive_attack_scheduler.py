"""
Adaptive Attack Scheduler for Quality-Diversity Algorithms
Implements epsilon creep and sigma ramping with safety catches
"""
import numpy as np
from typing import List, Dict, Optional
from ribs.emitters import EmitterBase


class AdaptiveAttackScheduler:
    """
    Manages adaptive attack parameter adjustments to overcome stagnation.
    
    Uses incremental epsilon creep and controlled sigma ramping with safety limits
    to break through optimization plateaus while maintaining search quality.
    """
    
    def __init__(
        self,
        emitters: List[EmitterBase],
        perturbation_generator,
        baseline_sigma: float = 0.02,
        baseline_epsilon: float = 0.12,
        stagnation_threshold: int = 15,
        sigma_multiplier: float = 1.5,
        epsilon_increment: float = 0.01,
        epsilon_max: float = 0.20,
        sigma_max: float = 0.10,
        sigma_reset: float = 0.05,
        epsilon_decrease: float = 0.005
    ):
        """
        Initialize Adaptive Attack Scheduler
        
        Args:
            emitters: List of pyribs emitters to manage
            perturbation_generator: PerturbationGenerator instance to adjust epsilon
            baseline_sigma: Initial/baseline sigma value (default: 0.02)
            baseline_epsilon: Initial/baseline epsilon value (default: 0.12)
            stagnation_threshold: Iterations without progress before intervention (default: 15)
            sigma_multiplier: Factor to multiply sigma by when stagnating (default: 1.5)
            epsilon_increment: Amount to increment epsilon by (default: 0.01)
            epsilon_max: Maximum epsilon limit (default: 0.20)
            sigma_max: Maximum sigma before safety reset (default: 0.10)
            sigma_reset: Value to reset sigma to when exceeding max (default: 0.05)
            epsilon_decrease: Amount to decrease epsilon on progress (default: 0.005)
        """
        self.emitters = emitters
        self.pert_gen = perturbation_generator
        
        # Baseline values
        self.baseline_sigma = baseline_sigma
        self.baseline_epsilon = baseline_epsilon
        
        # Adaptive parameters
        self.stagnation_threshold = stagnation_threshold
        self.sigma_multiplier = sigma_multiplier
        self.epsilon_increment = epsilon_increment
        self.epsilon_max = epsilon_max
        self.sigma_max = sigma_max
        self.sigma_reset = sigma_reset
        self.epsilon_decrease = epsilon_decrease
        
        # State tracking
        self.current_sigma = baseline_sigma
        self.current_epsilon = baseline_epsilon
        self.last_num_elites = 0
        self.stagnation_counter = 0
        self.total_boosts = 0
        self.total_resets = 0
        
        # History for logging
        self.history = {
            'iteration': [],
            'num_elites': [],
            'sigma': [],
            'epsilon': [],
            'stagnation_counter': [],
            'event': []  # 'none', 'boost', 'reset', 'safety_catch'
        }
        
    def step(self, iteration: int, num_elites: int) -> Dict[str, any]:
        """
        Update scheduler state based on current archive status.
        Call this after scheduler.tell() in the main loop.
        
        Args:
            iteration: Current iteration number
            num_elites: Current number of elites in archive
            
        Returns:
            Dictionary with current status and any changes made
        """
        status = {
            'changed': False,
            'event': 'none',
            'sigma': self.current_sigma,
            'epsilon': self.current_epsilon,
            'stagnation_counter': self.stagnation_counter
        }
        
        # Check for progress
        if num_elites > self.last_num_elites:
            # Progress detected!
            event = self._handle_progress()
            status['changed'] = True
            status['event'] = event
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            
        else:
            # No progress - increment stagnation counter
            self.stagnation_counter += 1
            
            # Check if we need to boost
            if self.stagnation_counter >= self.stagnation_threshold:
                event = self._apply_boost()
                status['changed'] = True
                status['event'] = event
                
                # Reset counter after boost
                self.stagnation_counter = 0
        
        # Update last count
        self.last_num_elites = num_elites
        
        # Log history
        self.history['iteration'].append(iteration)
        self.history['num_elites'].append(num_elites)
        self.history['sigma'].append(self.current_sigma)
        self.history['epsilon'].append(self.current_epsilon)
        self.history['stagnation_counter'].append(self.stagnation_counter)
        self.history['event'].append(status['event'])
        
        status['sigma'] = self.current_sigma
        status['epsilon'] = self.current_epsilon
        
        return status
    
    def _apply_boost(self) -> str:
        """Apply boost to sigma and epsilon when stagnation detected"""
        
        # Epsilon Creep: Increment by fixed amount
        old_epsilon = self.current_epsilon
        self.current_epsilon = min(self.current_epsilon + self.epsilon_increment, self.epsilon_max)
        
        # Sigma Ramp: Multiply by factor
        old_sigma = self.current_sigma
        self.current_sigma = self.current_sigma * self.sigma_multiplier
        
        # Safety Catch: If sigma exceeds max, reset to intermediate value
        if self.current_sigma > self.sigma_max:
            self.current_sigma = self.sigma_reset
            event = 'safety_catch'
            print(f"\n  ⚠️  [Adaptive Attack] Sigma safety catch triggered!")
            print(f"     Sigma exceeded {self.sigma_max:.4f}, resetting to {self.sigma_reset:.4f}")
            print(f"     Epsilon: {old_epsilon:.4f} → {self.current_epsilon:.4f}")
        else:
            event = 'boost'
            print(f"\n  🚀 [Adaptive Attack] Stagnation detected after {self.stagnation_threshold} iterations!")
            print(f"     Epsilon Creep: {old_epsilon:.4f} → {self.current_epsilon:.4f} (+{self.epsilon_increment:.4f})")
            print(f"     Sigma Ramp: {old_sigma:.4f} → {self.current_sigma:.4f} (×{self.sigma_multiplier:.2f})")
        
        # Update emitters
        self._update_emitter_sigma(self.current_sigma)
        
        # Update perturbation generator
        self.pert_gen.epsilon = self.current_epsilon
        
        self.total_boosts += 1
        return event
    
    def _handle_progress(self) -> str:
        """Handle progress detection: reset sigma, slightly decrease epsilon"""
        
        old_sigma = self.current_sigma
        old_epsilon = self.current_epsilon
        
        # Reset sigma to baseline for fine-grained exploitation
        self.current_sigma = self.baseline_sigma
        
        # Slightly decrease epsilon to maintain breakthrough
        # (Don't fully reset - keep some of the gained ground)
        self.current_epsilon = max(
            self.baseline_epsilon,
            self.current_epsilon - self.epsilon_decrease
        )
        
        print(f"\n  🔄 [Adaptive Attack] Progress detected! New elite found!")
        print(f"     Sigma Reset: {old_sigma:.4f} → {self.current_sigma:.4f} (baseline)")
        print(f"     Epsilon Adjust: {old_epsilon:.4f} → {self.current_epsilon:.4f} (-{self.epsilon_decrease:.4f})")
        
        # Update emitters
        self._update_emitter_sigma(self.current_sigma)
        
        # Update perturbation generator
        self.pert_gen.epsilon = self.current_epsilon
        
        self.total_resets += 1
        return 'reset'
    
    def _update_emitter_sigma(self, new_sigma: float):
        """Update sigma for all emitters - handles different emitter types"""
        for emitter in self.emitters:
            # Different emitter types have different sigma attributes
            if hasattr(emitter, '_sigma'):
                # GaussianEmitter uses _sigma
                if isinstance(emitter._sigma, np.ndarray):
                    emitter._sigma = np.full_like(emitter._sigma, new_sigma)
                else:
                    emitter._sigma = new_sigma
            elif hasattr(emitter, 'sigma'):
                # Some emitters might use sigma directly
                emitter.sigma = new_sigma
            
            # For ES emitters with optimizers
            if hasattr(emitter, '_opt'):
                if hasattr(emitter._opt, 'sigma0'):
                    emitter._opt.sigma0 = new_sigma
                elif hasattr(emitter._opt, '_sigma0'):
                    emitter._opt._sigma0 = new_sigma
            
            # IsoLineEmitter
            if hasattr(emitter, '_iso_sigma'):
                emitter._iso_sigma = new_sigma
    
    def get_history(self) -> Dict[str, List]:
        """Get complete history of adaptive adjustments"""
        return self.history
    
    def get_current_state(self) -> Dict[str, any]:
        """Get current scheduler state"""
        return {
            'sigma': self.current_sigma,
            'epsilon': self.current_epsilon,
            'stagnation_counter': self.stagnation_counter,
            'last_num_elites': self.last_num_elites,
            'total_boosts': self.total_boosts,
            'total_resets': self.total_resets
        }
    
    def save_history(self, save_path: str):
        """Save history to file"""
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self.history, f)
        print(f"  ✓ Saved adaptive attack history to {save_path}")
    
    def get_summary(self) -> str:
        """Get summary statistics"""
        summary = f"\nAdaptive Attack Scheduler Summary:\n"
        summary += f"  Total Boosts: {self.total_boosts}\n"
        summary += f"  Total Resets: {self.total_resets}\n"
        summary += f"  Final Sigma: {self.current_sigma:.4f}\n"
        summary += f"  Final Epsilon: {self.current_epsilon:.4f}\n"
        summary += f"  Baseline Sigma: {self.baseline_sigma:.4f}\n"
        summary += f"  Baseline Epsilon: {self.baseline_epsilon:.4f}\n"
        summary += f"  Max Sigma Limit: {self.sigma_max:.4f}\n"
        summary += f"  Max Epsilon Limit: {self.epsilon_max:.4f}\n"
        return summary
