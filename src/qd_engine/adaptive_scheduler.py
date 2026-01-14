"""
Resolution-Aware Adaptive Scheduler for MAP-Elites
Dynamically adjusts sigma and epsilon based on discovery stagnation
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque


class AdaptiveScheduler:
    """
    Adaptive parameter scheduler for CMA-ME and related algorithms
    
    Adjusts sigma0 and epsilon based on discovery rate to escape local optima
    """
    
    def __init__(
        self,
        base_scheduler,
        bc_ranges: List[Tuple[float, float]],
        grid_dims: Tuple[int, int],
        initial_epsilon: float = 0.05,
        max_epsilon: float = 0.20,
        epsilon_creep: float = 0.01,
        sigma_burst_multiplier: float = 1.5,
        max_sigma: float = 0.10,
        min_sigma: float = 0.05,
        stagnation_window: int = 15,
        verbose: bool = True
    ):
        """
        Initialize adaptive scheduler wrapper
        
        Args:
            base_scheduler: Underlying pyribs scheduler (CMA-ME, etc.)
            bc_ranges: List of (min, max) tuples for each behavior dimension
            grid_dims: Grid dimensions (height, width)
            initial_epsilon: Starting epsilon value
            max_epsilon: Maximum epsilon allowed
            epsilon_creep: Increment for epsilon during stagnation
            sigma_burst_multiplier: Multiplier for sigma burst
            max_sigma: Maximum sigma before reset
            min_sigma: Reset sigma value
            stagnation_window: Iterations without improvement to trigger adaptation
            verbose: Print adaptation events
        """
        self.scheduler = base_scheduler
        self.bc_ranges = bc_ranges
        self.grid_dims = grid_dims
        self.initial_epsilon = initial_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon_creep = epsilon_creep
        self.sigma_burst_multiplier = sigma_burst_multiplier
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.stagnation_window = stagnation_window
        self.verbose = verbose
        
        # Calculate base sigma from grid resolution
        self.base_sigma = self._calculate_base_sigma()
        
        # Tracking state
        self.current_epsilon = initial_epsilon
        self.current_sigma = self.base_sigma
        self.previous_num_elites = 0
        self.stagnation_counter = 0
        self.successful_epsilon = initial_epsilon
        
        # History for analytics
        self.epsilon_history = []
        self.sigma_history = []
        self.adaptation_events = []
        
        # Set initial sigma in base scheduler
        self._update_scheduler_sigma(self.current_sigma)
        
    def _calculate_base_sigma(self) -> float:
        """
        Calculate base sigma from BC ranges and grid resolution
        
        Formula: base_sigma = max(bin_width_bc1, bin_width_bc2) * 1.5
        """
        bin_widths = []
        for (bc_min, bc_max), grid_size in zip(self.bc_ranges, self.grid_dims):
            bin_width = (bc_max - bc_min) / grid_size
            bin_widths.append(bin_width)
        
        base_sigma = max(bin_widths) * 1.5
        
        if self.verbose:
            print(f"\n[AdaptiveScheduler] Base Sigma Calculation:")
            print(f"  BC Ranges: {self.bc_ranges}")
            print(f"  Grid Dims: {self.grid_dims}")
            print(f"  Bin Widths: {[f'{w:.4f}' for w in bin_widths]}")
            print(f"  Base Sigma: {base_sigma:.4f}")
        
        return base_sigma
    
    def ask(self) -> np.ndarray:
        """Generate new solutions from base scheduler"""
        return self.scheduler.ask()
    
    def tell(
        self,
        objectives: np.ndarray,
        measures: np.ndarray,
        **fields
    ):
        """
        Update scheduler and check for stagnation-based adaptation
        
        Args:
            objectives: Fitness values
            measures: Behavior characteristics
            **fields: Additional fields for scheduler
        """
        # Tell base scheduler
        self.scheduler.tell(objectives, measures, **fields)
        
        # Get current archive stats
        current_num_elites = len(self.scheduler.archive)
        
        # Check if new elites were discovered
        if current_num_elites > self.previous_num_elites:
            # Discovery! Reset stagnation and restore successful parameters
            self._handle_discovery(current_num_elites)
        else:
            # No new elites - increment stagnation counter
            self.stagnation_counter += 1
            
            # Check if stagnation threshold reached
            if self.stagnation_counter >= self.stagnation_window:
                self._handle_stagnation()
        
        # Update tracking
        self.previous_num_elites = current_num_elites
        self.epsilon_history.append(self.current_epsilon)
        self.sigma_history.append(self.current_sigma)
    
    def _handle_discovery(self, current_num_elites: int):
        """Handle successful elite discovery"""
        new_elites = current_num_elites - self.previous_num_elites
        
        if self.verbose and self.stagnation_counter > 0:
            print(f"\n[AdaptiveScheduler] 🎯 Discovery Event!")
            print(f"  New Elites: {new_elites}")
            print(f"  Resetting sigma to base: {self.base_sigma:.4f}")
            print(f"  Restoring epsilon to: {self.successful_epsilon:.4f}")
        
        # Reset to successful parameters
        self.current_sigma = self.base_sigma
        self.current_epsilon = self.successful_epsilon
        self.stagnation_counter = 0
        
        # Update scheduler
        self._update_scheduler_sigma(self.current_sigma)
        
        # Log event
        self.adaptation_events.append({
            'type': 'discovery_reset',
            'iteration': len(self.sigma_history),
            'new_elites': new_elites,
            'sigma': self.current_sigma,
            'epsilon': self.current_epsilon
        })
    
    def _handle_stagnation(self):
        """Handle stagnation - increase epsilon and burst sigma"""
        old_epsilon = self.current_epsilon
        old_sigma = self.current_sigma
        
        # Epsilon creep (up to max)
        self.current_epsilon = min(
            self.current_epsilon + self.epsilon_creep,
            self.max_epsilon
        )
        
        # Sigma burst (with reset if exceeds max)
        new_sigma = self.current_sigma * self.sigma_burst_multiplier
        if new_sigma > self.max_sigma:
            self.current_sigma = self.min_sigma  # Reset to base
            reset_triggered = True
        else:
            self.current_sigma = new_sigma
            reset_triggered = False
        
        if self.verbose:
            print(f"\n[AdaptiveScheduler] 📉 Stagnation Detected!")
            print(f"  No new elites for {self.stagnation_window} iterations")
            print(f"  Epsilon: {old_epsilon:.4f} → {self.current_epsilon:.4f}")
            if reset_triggered:
                print(f"  Sigma: {old_sigma:.4f} → {self.current_sigma:.4f} (RESET)")
            else:
                print(f"  Sigma: {old_sigma:.4f} → {self.current_sigma:.4f} (×{self.sigma_burst_multiplier})")
        
        # Update scheduler
        self._update_scheduler_sigma(self.current_sigma)
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        # Log event
        self.adaptation_events.append({
            'type': 'stagnation_adaptation',
            'iteration': len(self.sigma_history),
            'old_epsilon': old_epsilon,
            'new_epsilon': self.current_epsilon,
            'old_sigma': old_sigma,
            'new_sigma': self.current_sigma,
            'sigma_reset': reset_triggered
        })
    
    def _update_scheduler_sigma(self, new_sigma: float):
        """Update sigma in underlying scheduler"""
        # Different schedulers have different sigma access patterns
        if hasattr(self.scheduler, 'sigma0'):
            self.scheduler.sigma0 = new_sigma
        elif hasattr(self.scheduler, 'emitters'):
            # For schedulers with emitter lists
            for emitter in self.scheduler.emitters:
                if hasattr(emitter, 'sigma0'):
                    emitter.sigma0 = new_sigma
                elif hasattr(emitter, 'sigma'):
                    emitter.sigma = new_sigma
        elif hasattr(self.scheduler, 'sigma'):
            self.scheduler.sigma = new_sigma
    
    def get_status_string(self) -> str:
        """Get current status as formatted string"""
        return (f"Sigma={self.current_sigma:.4f} | "
                f"Epsilon={self.current_epsilon:.4f} | "
                f"Stagnation={self.stagnation_counter}/{self.stagnation_window}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        epsilon_changes = sum(1 for e in self.adaptation_events if e['type'] == 'stagnation_adaptation')
        sigma_resets = sum(1 for e in self.adaptation_events if e.get('sigma_reset', False))
        discoveries = sum(1 for e in self.adaptation_events if e['type'] == 'discovery_reset')
        
        return {
            'final_sigma': self.current_sigma,
            'final_epsilon': self.current_epsilon,
            'base_sigma': self.base_sigma,
            'total_adaptations': len(self.adaptation_events),
            'epsilon_increases': epsilon_changes,
            'sigma_resets': sigma_resets,
            'discovery_resets': discoveries,
            'epsilon_history': self.epsilon_history,
            'sigma_history': self.sigma_history,
            'adaptation_events': self.adaptation_events
        }
    
    def save_history(self, filepath: str):
        """Save adaptation history to file"""
        stats = self.get_statistics()
        np.savez(
            filepath,
            epsilon_history=np.array(self.epsilon_history),
            sigma_history=np.array(self.sigma_history),
            adaptation_events=self.adaptation_events,
            **{k: v for k, v in stats.items() if not isinstance(v, list)}
        )
    
    @property
    def archive(self):
        """Access underlying archive"""
        return self.scheduler.archive
    
    def __getattr__(self, name):
        """Delegate unknown attributes to base scheduler"""
        return getattr(self.scheduler, name)
