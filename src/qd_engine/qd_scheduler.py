"""
QD Scheduler for managing emitters and evolution loop
"""
import sys
import os
# Use local pyribs if available
if os.path.exists('/root/ICAT/pyribs') and '/root/ICAT/pyribs' not in sys.path:
    sys.path.insert(0, '/root/ICAT/pyribs')

import numpy as np
from ribs.schedulers import Scheduler
from typing import List, Dict, Any


class QDScheduler:
    """Wrapper for Pyribs Scheduler with custom functionality"""
    
    def __init__(self, archive, emitters):
        """
        Initialize QD Scheduler
        
        Args:
            archive: QDArchive instance
            emitters: List of emitter instances
        """
        self.archive = archive
        self.emitters = emitters
        
        # Create scheduler
        self.scheduler = Scheduler(
            archive.archive,
            emitters
        )
        
        self.iteration = 0
        self.history = {
            'qd_score': [],
            'coverage': [],
            'max_fitness': [],
            'mean_fitness': [],
            'num_elites': []
        }
        
    def ask(self):
        """
        Request new solutions from emitters
        
        Returns:
            Array of solutions to evaluate [batch_size, solution_dim]
        """
        return self.scheduler.ask()
    
    def tell(self, objectives: np.ndarray, measures: np.ndarray, **kwargs):
        """
        Return evaluation results to scheduler
        
        Args:
            objectives: Fitness values [batch_size]
            measures: Behavior characteristics [batch_size, n_measures]
            **kwargs: Additional metadata (e.g., jacobian for CMA-MEGA)
        """
        self.scheduler.tell(objectives, measures, **kwargs)
        self.iteration += 1
        
        # Update history
        stats = self.archive.get_stats()
        for key in self.history.keys():
            self.history[key].append(stats[key])
    
    def tell_with_jacobian(self, objectives: np.ndarray, measures: np.ndarray,
                          jacobian: np.ndarray):
        """
        Tell with Jacobian for gradient-assisted algorithms (CMA-MEGA, CMA-MEGA)
        
        Args:
            objectives: Fitness values [batch_size]
            measures: Behavior characteristics [batch_size, n_measures]
            jacobian: Gradient information [batch_size, n_measures, solution_dim]
        """
        self.scheduler.tell(objectives, measures, jacobian=jacobian)
        self.iteration += 1
        
        # Update history
        stats = self.archive.get_stats()
        for key in self.history.keys():
            self.history[key].append(stats[key])
    
    def get_iteration(self):
        """Get current iteration number"""
        return self.iteration
    
    def get_history(self):
        """Get training history"""
        return self.history
    
    def get_stats(self):
        """Get current archive statistics"""
        return self.archive.get_stats()
