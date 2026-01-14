"""
Visual Stealth Archive - Prioritizes minimal perturbation within behavior bins
"""
import numpy as np
from typing import Dict, Any, Optional
from ribs.archives import GridArchive


class VisualStealthArchive:
    """
    Custom archive wrapper that implements "Niche Optimization" for visual stealth.
    
    When multiple solutions fall into the same behavior cell, this archive:
    - Prioritizes solutions with HIGHER fitness (better attack)
    - If fitness is equal, prioritizes solutions with LOWER BC1 (smaller L-inf perturbation)
    
    This ensures we always keep the most "stealthy" attack in each niche.
    """
    
    def __init__(self, 
                 solution_dim: int,
                 grid_dims: tuple = (50, 50),
                 ranges: list = None,
                 learning_rate: float = 1.0,
                 threshold_min: float = -np.inf):
        """
        Initialize Visual Stealth Archive
        
        Args:
            solution_dim: Dimensionality of solutions
            grid_dims: Dimensions of MAP-Elites grid
            ranges: Ranges for behavior characteristics [(min1, max1), (min2, max2)]
            learning_rate: Learning rate for archive (1.0 for standard MAP-Elites)
            threshold_min: Minimum threshold for accepting solutions
        """
        self.solution_dim = solution_dim
        self.grid_dims = grid_dims
        self.ranges = ranges if ranges is not None else [(0.0, 0.1), (0.0, 1.0)]
        
        # Create underlying GridArchive
        self.archive = GridArchive(
            solution_dim=solution_dim,
            dims=grid_dims,
            ranges=self.ranges,
            learning_rate=learning_rate,
            threshold_min=threshold_min
        )
        
        # Track replacements for logging
        self.replacement_stats = {
            'total_adds': 0,
            'fitness_improvements': 0,
            'stealth_improvements': 0,
            'new_cells': 0,
            'rejections': 0
        }
    
    def add(self, solutions: np.ndarray, objectives: np.ndarray, 
            measures: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Add solutions with visual stealth optimization
        
        For each solution:
        1. Check if the target cell is occupied
        2. If occupied:
           - Replace if new_fitness > current_fitness (attack improvement)
           - Replace if new_fitness == current_fitness AND new_bc1 < current_bc1 (stealth improvement)
        3. If not occupied, add normally
        
        Args:
            solutions: Array of solutions [batch_size, solution_dim]
            objectives: Array of fitness values [batch_size]
            measures: Array of behavior characteristics [batch_size, n_measures]
            
        Returns:
            Dictionary with add results and statistics
        """
        batch_size = len(solutions)
        
        for i in range(batch_size):
            solution = solutions[i]
            objective = objectives[i]
            measure = measures[i]
            bc1 = measure[0]  # L-inf norm (perturbation magnitude)
            
            # Check if cell is occupied using retrieve_single
            occupied, current_elite_data = self.archive.retrieve_single(measure)
            
            if occupied:
                # Get current elite data
                current_objective = current_elite_data['objective']
                current_bc1 = current_elite_data['measures'][0]
                
                # Decision logic: "Efficiency over Brutality"
                should_replace = False
                replacement_reason = None
                
                # Criterion 1: Better fitness (stronger attack)
                if objective > current_objective:
                    should_replace = True
                    replacement_reason = 'fitness'
                    self.replacement_stats['fitness_improvements'] += 1
                
                # Criterion 2: Equal fitness but lower perturbation (stealthier)
                elif np.isclose(objective, current_objective, rtol=1e-6, atol=1e-8):
                    if bc1 < current_bc1:
                        should_replace = True
                        replacement_reason = 'stealth'
                        self.replacement_stats['stealth_improvements'] += 1
                
                if should_replace:
                    # Add to archive (will replace existing elite)
                    self.archive.add_single(solution, objective, measure, **kwargs)
                else:
                    # Reject (current elite is better)
                    self.replacement_stats['rejections'] += 1
            else:
                # Cell is empty - add normally
                self.archive.add_single(solution, objective, measure, **kwargs)
                self.replacement_stats['new_cells'] += 1
            
            self.replacement_stats['total_adds'] += 1
        
        # Return standard add result
        return {
            'n_added': batch_size,
            'replacement_stats': self.replacement_stats.copy()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get archive statistics including replacement stats"""
        stats = self.archive.stats
        
        base_stats = {
            'qd_score': stats.qd_score,
            'coverage': stats.coverage,
            'max_fitness': stats.obj_max,
            'mean_fitness': stats.obj_mean,
            'num_elites': len(self.archive)
        }
        
        # Add replacement statistics
        base_stats.update({
            'total_adds': self.replacement_stats['total_adds'],
            'fitness_improvements': self.replacement_stats['fitness_improvements'],
            'stealth_improvements': self.replacement_stats['stealth_improvements'],
            'new_cells': self.replacement_stats['new_cells'],
            'rejections': self.replacement_stats['rejections']
        })
        
        return base_stats
    
    def get_replacement_summary(self) -> str:
        """Get a human-readable summary of replacement statistics"""
        stats = self.replacement_stats
        total = stats['total_adds']
        
        if total == 0:
            return "No solutions added yet"
        
        summary = (
            f"Archive Replacement Statistics:\n"
            f"  Total Additions Attempted: {total}\n"
            f"  New Cells Filled:          {stats['new_cells']} ({stats['new_cells']/total*100:.1f}%)\n"
            f"  Fitness Improvements:      {stats['fitness_improvements']} ({stats['fitness_improvements']/total*100:.1f}%)\n"
            f"  Stealth Improvements:      {stats['stealth_improvements']} ({stats['stealth_improvements']/total*100:.1f}%)\n"
            f"  Rejections (worse):        {stats['rejections']} ({stats['rejections']/total*100:.1f}%)\n"
        )
        
        return summary
    
    def get_best_elite(self):
        """Get elite with highest objective value"""
        if len(self.archive) == 0:
            return None
        
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
    
    def save(self, filepath: str):
        """Save archive to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'archive': self.archive,
                'replacement_stats': self.replacement_stats
            }, f)
        print(f"Visual Stealth Archive saved to {filepath}")
    
    def load(self, filepath: str):
        """Load archive from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.archive = data['archive']
            self.replacement_stats = data.get('replacement_stats', self.replacement_stats)
        print(f"Visual Stealth Archive loaded from {filepath}")
    
    def get_heatmap_data(self):
        """Get data for heatmap visualization"""
        df = self.archive.as_pandas()
        if len(df) == 0:
            return np.zeros(self.grid_dims)
        
        grid = np.full(self.grid_dims, np.nan)
        
        for _, row in df.iterrows():
            idx = tuple(row['index_0':'index_{}'.format(len(self.grid_dims)-1)].astype(int))
            grid[idx] = row['objective']
        
        return grid
    
    def __len__(self):
        """Return number of elites in archive"""
        return len(self.archive)
