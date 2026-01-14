"""
QD Archive wrapper for managing the MAP-Elites grid
"""
import sys
import os
# Use local pyribs if available
if os.path.exists('/root/ICAT/pyribs') and '/root/ICAT/pyribs' not in sys.path:
    sys.path.insert(0, '/root/ICAT/pyribs')

import numpy as np
from ribs.archives import GridArchive
from typing import Tuple, List


class QDArchive:
    """Wrapper for Pyribs GridArchive with custom functionality"""
    
    def __init__(self, 
                 solution_dim: int,
                 grid_dims: Tuple[int, ...] = (50, 50),
                 ranges: List[Tuple[float, float]] = None,
                 learning_rate: float = 1.0,
                 threshold_min: float = -np.inf):
        """
        Initialize QD Archive
        
        Args:
            solution_dim: Dimensionality of solutions (e.g., flattened perturbation)
            grid_dims: Dimensions of the MAP-Elites grid
            ranges: Ranges for each behavior characteristic (BC)
            learning_rate: Learning rate for archive updates (use 1.0 for standard MAP-Elites)
            threshold_min: Minimum threshold for accepting solutions (use -np.inf with learning_rate=1.0)
        """
        self.solution_dim = solution_dim
        self.grid_dims = grid_dims
        
        # Default ranges if not provided
        if ranges is None:
            ranges = [(0.0, 0.1), (0.0, 1.0)]  # L-inf norm, Spectral energy
        
        self.ranges = ranges
        
        # Create the archive
        self.archive = GridArchive(
            solution_dim=solution_dim,
            dims=grid_dims,
            ranges=ranges,
            learning_rate=learning_rate,
            threshold_min=threshold_min
        )
        
    def add(self, solutions: np.ndarray, objectives: np.ndarray, 
            measures: np.ndarray, **kwargs):
        """
        Add solutions to the archive
        
        Args:
            solutions: Array of solutions [batch_size, solution_dim]
            objectives: Array of objective values [batch_size]
            measures: Array of behavior characteristics [batch_size, n_measures]
            **kwargs: Additional metadata
        """
        return self.archive.add(solutions, objectives, measures, **kwargs)
    
    def get_stats(self):
        """Get archive statistics"""
        stats = self.archive.stats
        return {
            'qd_score': stats.qd_score,
            'coverage': stats.coverage,
            'max_fitness': stats.obj_max,
            'mean_fitness': stats.obj_mean,
            'num_elites': len(self.archive)
        }
    
    def get_best_elite(self):
        """Get the elite with highest objective value"""
        if len(self.archive) == 0:
            return None
        
        # Use iterate() to get all elites (returns dicts)
        best_obj = float('-inf')
        best_elite = None
        
        for elite in self.archive:
            if elite['objective'] > best_obj:
                best_obj = elite['objective']
                best_elite = {
                    'solution': elite['solution'],
                    'objective': elite['objective'],
                    'measures': elite['measures']
                }
        
        return best_elite
    
    def get_random_elites(self, n: int = 1):
        """
        Sample random elites from the archive
        
        Args:
            n: Number of elites to sample
            
        Returns:
            Array of solutions [n, solution_dim]
        """
        import numpy as np
        elites = list(self.archive)
        if len(elites) == 0:
            return np.array([])
        
        sampled = np.random.choice(len(elites), size=min(n, len(elites)), replace=False)
        return np.array([elites[i]['solution'] for i in sampled])
    
    def save(self, filepath: str):
        """Save archive to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.archive, f)
        print(f"Archive saved to {filepath}")
    
    def load(self, filepath: str):
        """Load archive from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            self.archive = pickle.load(f)
        print(f"Archive loaded from {filepath}")
    
    def get_heatmap_data(self):
        """
        Get data for heatmap visualization
        
        Returns:
            Grid of objective values
        """
        df = self.archive.as_pandas()
        if len(df) == 0:
            return np.zeros(self.grid_dims)
        
        # Create grid
        grid = np.full(self.grid_dims, np.nan)
        
        for _, row in df.iterrows():
            idx = tuple(row['index_0':'index_{}'.format(len(self.grid_dims)-1)].astype(int))
            grid[idx] = row['objective']
        
        return grid
    
    def __len__(self):
        """Return number of elites in archive"""
        return len(self.archive)
