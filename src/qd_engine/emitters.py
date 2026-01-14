"""
Emitter creation utilities for different QD algorithms
"""
import sys
import os
# Use local pyribs if available
if os.path.exists('/root/ICAT/pyribs') and '/root/ICAT/pyribs' not in sys.path:
    sys.path.insert(0, '/root/ICAT/pyribs')

from ribs.emitters import (
    EvolutionStrategyEmitter,
    GaussianEmitter,
    IsoLineEmitter
)
import numpy as np
from typing import List


def create_emitters(archive, 
                   algorithm: str = 'cma_me',
                   num_emitters: int = 5,
                   sigma0: float = 0.02,
                   **kwargs):
    """
    Create emitters for QD algorithms
    
    Args:
        archive: Archive instance (GridArchive)
        algorithm: Algorithm type ('map_elites', 'cma_me', 'cma_mae', 'cma_mega')
        num_emitters: Number of emitters to create
        sigma0: Initial step size for evolution strategies
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        List of emitter instances
    """
    emitters = []
    
    if algorithm.lower() == 'map_elites':
        # Standard MAP-Elites with Gaussian mutation
        for i in range(num_emitters):
            emitters.append(
                GaussianEmitter(
                    archive,
                    x0=kwargs.get('x0', np.zeros(archive.solution_dim)),
                    sigma0=sigma0,
                    batch_size=kwargs.get('batch_size', 36)
                )
            )
    
    elif algorithm.lower() == 'cma_me':
        # CMA-ME: Covariance Matrix Adaptation MAP-Elites
        for i in range(num_emitters):
            emitters.append(
                EvolutionStrategyEmitter(
                    archive,
                    x0=kwargs.get('x0', np.zeros(archive.solution_dim)),
                    sigma0=sigma0,
                    ranker='obj',  # Rank by objective only
                    batch_size=kwargs.get('batch_size', 36),
                    es='cma_es'
                )
            )
    
    elif algorithm.lower() == 'cma_mae':
        # CMA-MAE: Improvement-based ranking
        for i in range(num_emitters):
            emitters.append(
                EvolutionStrategyEmitter(
                    archive,
                    x0=kwargs.get('x0', np.zeros(archive.solution_dim)),
                    sigma0=sigma0,
                    ranker='imp',  # Rank by improvement
                    batch_size=kwargs.get('batch_size', 36),
                    es='cma_es'
                )
            )
    
    elif algorithm.lower() == 'cma_mega':
        # CMA-MEGA: Gradient-assisted MAP-Elites
        # Note: Requires Jacobian information in tell()
        for i in range(num_emitters):
            emitters.append(
                EvolutionStrategyEmitter(
                    archive,
                    x0=kwargs.get('x0', np.zeros(archive.solution_dim)),
                    sigma0=sigma0,
                    ranker='2imp',  # Rank by improvement + gradient
                    batch_size=kwargs.get('batch_size', 36),
                    es='cma_es'
                )
            )
    
    elif algorithm.lower() == 'isoline':
        # IsoLine emitters for diversity
        for i in range(num_emitters):
            emitters.append(
                IsoLineEmitter(
                    archive,
                    x0=kwargs.get('x0', np.zeros(archive.solution_dim)),
                    sigma0=sigma0,
                    batch_size=kwargs.get('batch_size', 36),
                    iso_sigma=kwargs.get('iso_sigma', 0.01),
                    line_sigma=kwargs.get('line_sigma', 0.2)
                )
            )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"Created {num_emitters} emitters for {algorithm.upper()}")
    return emitters


def create_mixed_emitters(archive, sigma0: float = 0.01, **kwargs):
    """
    Create a mix of different emitter types for hybrid exploration
    
    Args:
        archive: Archive instance
        sigma0: Initial step size
        **kwargs: Additional parameters
        
    Returns:
        List of mixed emitter instances
    """
    emitters = []
    
    # Add CMA-ES emitters
    for _ in range(3):
        emitters.append(
            EvolutionStrategyEmitter(
                archive,
                x0=kwargs.get('x0', np.zeros(archive.solution_dim)),
                sigma0=sigma0,
                ranker='obj',
                batch_size=kwargs.get('batch_size', 36),
                es='cma_es'
            )
        )
    
    # Add IsoLine emitters
    for _ in range(2):
        emitters.append(
            IsoLineEmitter(
                archive,
                x0=kwargs.get('x0', np.zeros(archive.solution_dim)),
                sigma0=sigma0,
                batch_size=kwargs.get('batch_size', 36)
            )
        )
    
    print(f"Created mixed emitters: 3 CMA-ES + 2 IsoLine")
    return emitters
