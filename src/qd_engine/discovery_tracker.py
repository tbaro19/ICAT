"""
Discovery Rate Tracker for monitoring MAP-Elites convergence
"""
import numpy as np
from collections import deque
from typing import Optional, Dict, Any


class DiscoveryRateTracker:
    """
    Tracks the discovery rate (new elites per iteration) using a rolling window.
    Detects convergence/stagnation when the rate drops below a threshold.
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 convergence_threshold: float = 0.02,
                 consecutive_warnings: int = 2):
        """
        Initialize Discovery Rate Tracker
        
        Args:
            window_size: Size of rolling window for computing discovery rate
            convergence_threshold: Minimum acceptable discovery rate (new elites/iter)
            consecutive_warnings: Number of consecutive low-rate windows before warning
        """
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.consecutive_warnings = consecutive_warnings
        
        # Rolling window for elite counts
        self.elite_counts = deque(maxlen=window_size)
        
        # Tracking state
        self.current_discovery_rate = 0.0
        self.low_rate_counter = 0
        self.convergence_warned = False
        self.iteration = 0
        
        # History for logging
        self.history = {
            'iteration': [],
            'num_elites': [],
            'discovery_rate': [],
            'is_converged': []
        }
        
    def update(self, num_elites: int) -> Dict[str, Any]:
        """
        Update tracker with current elite count and check for convergence
        
        Args:
            num_elites: Current number of elites in archive
            
        Returns:
            Dictionary with current status:
                - discovery_rate: Current discovery rate (new elites/iter)
                - is_converged: Whether convergence warning should be triggered
                - message: Status message (if any)
        """
        self.iteration += 1
        self.elite_counts.append(num_elites)
        
        # Calculate discovery rate once we have enough data
        if len(self.elite_counts) >= 2:
            # Discovery rate = new elites per iteration over the window
            new_elites = self.elite_counts[-1] - self.elite_counts[0]
            self.current_discovery_rate = new_elites / len(self.elite_counts)
        else:
            self.current_discovery_rate = 0.0
        
        # Check for convergence only after filling the window
        is_converged = False
        message = None
        
        if len(self.elite_counts) == self.window_size:
            if self.current_discovery_rate < self.convergence_threshold:
                self.low_rate_counter += 1
                
                # Trigger warning after consecutive low-rate windows
                if self.low_rate_counter >= self.consecutive_warnings and not self.convergence_warned:
                    is_converged = True
                    self.convergence_warned = True
                    message = (
                        f"\n{'='*60}\n"
                        f"⚠️  CONVERGENCE WARNING: Search has plateaued!\n"
                        f"{'='*60}\n"
                        f"  Discovery Rate: {self.current_discovery_rate:.4f} (threshold: {self.convergence_threshold:.4f})\n"
                        f"  Window Size: {self.window_size} iterations\n"
                        f"  Consecutive Low-Rate Windows: {self.low_rate_counter}\n"
                        f"  \n"
                        f"  Recommendation: The archive is near saturation.\n"
                        f"  - Consider increasing epsilon or sigma for more exploration\n"
                        f"  - Or stop training if target coverage is reached\n"
                        f"{'='*60}\n"
                    )
            else:
                # Reset counter if rate improves
                self.low_rate_counter = 0
                self.convergence_warned = False
        
        # Log to history
        self.history['iteration'].append(self.iteration)
        self.history['num_elites'].append(num_elites)
        self.history['discovery_rate'].append(self.current_discovery_rate)
        self.history['is_converged'].append(is_converged)
        
        return {
            'discovery_rate': self.current_discovery_rate,
            'is_converged': is_converged,
            'message': message,
            'low_rate_counter': self.low_rate_counter,
            'window_filled': len(self.elite_counts) == self.window_size
        }
    
    def get_status_string(self) -> str:
        """
        Get a compact status string for logging
        
        Returns:
            Status string with discovery rate and convergence info
        """
        if len(self.elite_counts) < 2:
            return "Discovery: Warming up..."
        
        status = f"Discovery Rate: {self.current_discovery_rate:.4f}"
        
        if len(self.elite_counts) == self.window_size:
            if self.current_discovery_rate < self.convergence_threshold:
                status += f" (LOW {self.low_rate_counter}/{self.consecutive_warnings})"
            else:
                status += " (OK)"
        else:
            status += f" (window {len(self.elite_counts)}/{self.window_size})"
        
        return status
    
    def get_history(self) -> Dict[str, list]:
        """Get tracking history"""
        return self.history
    
    def reset(self):
        """Reset tracker state (useful for adaptive scheduler interventions)"""
        self.elite_counts.clear()
        self.current_discovery_rate = 0.0
        self.low_rate_counter = 0
        self.convergence_warned = False
        
    def save_history(self, filepath: str):
        """Save tracking history to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)
        print(f"Discovery tracker history saved to {filepath}")
