"""
Unified Attack Manager for Multi-Model Adversarial MAP-Elites
Orchestrates logit-based fitness, adaptive scheduling, and visual stealth optimization
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from .adaptive_scheduler import AdaptiveScheduler
from .visual_stealth_archive import VisualStealthArchive
from .discovery_tracker import DiscoveryRateTracker
from ..attack.logit_fitness import UnifiedLogitFitness
from ..utils.golden_elites import export_golden_elites


class UnifiedAttackManager:
    """
    Central orchestrator for unified adaptive adversarial attacks
    
    Integrates:
    - Multi-model logit-based fitness
    - Resolution-aware adaptive scheduling
    - Visual stealth archive optimization
    - Discovery rate tracking
    - Golden elite extraction
    """
    
    def __init__(
        self,
        vlm_model,
        archive: VisualStealthArchive,
        base_scheduler,
        pert_gen,
        measure_fn,
        target_image: np.ndarray,
        original_caption: str,
        groundtruth_caption: str,
        bc_ranges: List[tuple],
        grid_dims: tuple,
        use_logit_loss: bool = True,
        initial_epsilon: float = 0.05,
        device: str = 'cuda',
        chunk_size: int = 4,
        verbose: bool = True
    ):
        """
        Initialize unified attack manager
        
        Args:
            vlm_model: VLM model wrapper
            archive: Visual stealth archive instance
            base_scheduler: Base pyribs scheduler (CMA-ME, etc.)
            pert_gen: Perturbation generator
            measure_fn: Behavior measure function
            target_image: Clean image to attack [C, H, W]
            original_caption: Original generated caption
            groundtruth_caption: Ground-truth caption
            bc_ranges: BC dimension ranges for adaptive scheduler
            grid_dims: Grid dimensions (height, width)
            use_logit_loss: Use logit-based fitness (True) or similarity (False)
            initial_epsilon: Starting epsilon for adaptive scheduler
            device: Computation device
            chunk_size: Batch size for processing
            verbose: Print detailed logs
        """
        self.vlm_model = vlm_model
        self.archive = archive
        self.pert_gen = pert_gen
        self.measure_fn = measure_fn
        self.target_image = target_image
        self.original_caption = original_caption
        self.groundtruth_caption = groundtruth_caption
        self.device = device
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        # Initialize unified fitness function
        # Use smaller chunk size for logit-loss to avoid OOM
        fitness_chunk_size = 1 if use_logit_loss else chunk_size
        self.fitness_fn = UnifiedLogitFitness(
            use_logit_loss=use_logit_loss,
            device=device,
            chunk_size=fitness_chunk_size
        )
        
        # Wrap base scheduler with adaptive scheduler
        self.scheduler = AdaptiveScheduler(
            base_scheduler=base_scheduler,
            bc_ranges=bc_ranges,
            grid_dims=grid_dims,
            initial_epsilon=initial_epsilon,
            verbose=verbose
        )
        
        # Initialize discovery rate tracker
        self.discovery_tracker = DiscoveryRateTracker(
            window_size=50,
            convergence_threshold=0.02,
            consecutive_warnings=2
        )
        
        # Cache text embeddings for faster evaluation (if using similarity fitness)
        self.original_text_emb = None
        self.groundtruth_text_emb = None
        if not use_logit_loss:
            with torch.no_grad():
                self.original_text_emb = vlm_model.get_text_embeddings([original_caption])
                self.groundtruth_text_emb = vlm_model.get_text_embeddings([groundtruth_caption])
        
        # Statistics
        self.iteration_count = 0
        self.total_evaluations = 0
        
        # History tracking for visualizations
        self.qd_score_history = []
        self.coverage_history = []
        self.max_fitness_history = []
        self.mean_fitness_history = []
        
        if self.verbose:
            print("\n" + "="*60)
            print("🎯 Unified Attack Manager Initialized")
            print("="*60)
            print(f"Fitness Mode: {'Logit-Loss' if use_logit_loss else 'Similarity-Based'}")
            print(f"Grid Size: {grid_dims[0]}×{grid_dims[1]}")
            print(f"BC Ranges: {bc_ranges}")
            print(f"Base Sigma: {self.scheduler.base_sigma:.4f}")
            print(f"Initial Epsilon: {initial_epsilon:.4f}")
            print("="*60)
    
    def run_iteration(self) -> Dict[str, Any]:
        """
        Execute one MAP-Elites iteration
        
        Returns:
            Statistics dictionary for this iteration
        """
        # 1. Ask for new solutions
        solutions = self.scheduler.ask()
        
        # 2. Evaluate solutions in chunks
        all_objectives = []
        all_measures = []
        
        num_chunks = (len(solutions) + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(solutions))
            chunk_solutions = solutions[start_idx:end_idx]
            
            # Convert solutions to perturbations
            perturbations = self.pert_gen.solutions_to_perturbations(chunk_solutions)
            
            # Apply perturbations to target image
            perturbed_images = self.pert_gen.apply_perturbation(
                self.target_image, perturbations
            )
            
            # Compute fitness (logit-loss or similarity)
            objectives = self.fitness_fn.compute_fitness(
                original_captions=[self.original_caption] * len(perturbed_images),
                perturbed_images=perturbed_images,
                vlm_model=self.vlm_model,
                original_embeddings=self.original_text_emb
            )
            
            # Compute behavior characteristics
            measures = self.measure_fn.compute_measures(perturbations)
            
            all_objectives.extend(objectives)
            all_measures.append(measures)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine chunks
        all_objectives = np.array(all_objectives)
        all_measures = np.vstack(all_measures)
        
        # 3. Tell scheduler results (Visual Stealth Archive handles replacement)
        self.scheduler.tell(all_objectives, all_measures)
        
        # 4. Update discovery tracker and statistics
        current_num_elites = len(self.archive)
        discovery_status = self.discovery_tracker.update(current_num_elites)
        
        # 5. Update counters
        self.iteration_count += 1
        self.total_evaluations += len(solutions)
        
        # 6. Track history for visualizations
        archive_stats = self.archive.get_stats()
        self.qd_score_history.append(archive_stats['qd_score'])
        self.coverage_history.append(archive_stats['coverage'])
        self.max_fitness_history.append(archive_stats['max_fitness'])
        self.mean_fitness_history.append(archive_stats['mean_fitness'])
        
        # 7. Build iteration stats
        discovery_status = self.discovery_tracker.update(current_num_elites)
        stats = {
            'iteration': self.iteration_count,
            'num_elites': current_num_elites,
            'best_fitness': all_objectives.max(),
            'mean_fitness': all_objectives.mean(),
            'discovery_rate': self.discovery_tracker.current_discovery_rate,
            'sigma': self.scheduler.current_sigma,
            'epsilon': self.scheduler.current_epsilon,
            'stagnation': self.scheduler.stagnation_counter,
            'convergence_warning': discovery_status['is_converged']
        }
        
        return stats
    
    def run(self, num_iterations: int, checkpoint_interval: int = 100) -> Dict[str, Any]:
        """
        Run full MAP-Elites attack for specified iterations
        
        Args:
            num_iterations: Total iterations to run
            checkpoint_interval: Log interval
            
        Returns:
            Final statistics and results
        """
        if self.verbose:
            print(f"\n🚀 Starting {num_iterations} iterations...")
        
        for i in tqdm(range(num_iterations), desc="QD Attack"):
            stats = self.run_iteration()
            
            # Checkpoint logging
            if (i + 1) % checkpoint_interval == 0 or (i + 1) == num_iterations:
                self._log_checkpoint(stats)
        
        # Final statistics
        final_stats = self._compute_final_statistics()
        
        if self.verbose:
            self._print_final_summary(final_stats)
        
        return final_stats
    
    def _log_checkpoint(self, stats: Dict[str, Any]):
        """Log checkpoint information"""
        if not self.verbose:
            return
        
        print(f"\n{'='*60}")
        print(f"Iteration {stats['iteration']}")
        print(f"{'='*60}")
        print(f"Archive Coverage: {stats['num_elites']} elites")
        print(f"Best Fitness: {stats['best_fitness']:.4f}")
        print(f"Discovery Rate: {stats['discovery_rate']:.4f} new/iter")
        print(f"{self.scheduler.get_status_string()}")
        
        # Visual Stealth stats
        vs_stats = self.archive.get_stats()
        print(f"\nVisual Stealth Archive:")
        print(f"  Fitness Improvements: {vs_stats['fitness_improvements']}")
        print(f"  Stealth Improvements: {vs_stats['stealth_improvements']}")
        print(f"  Rejections: {vs_stats['rejections']}")
        
        # Convergence warning
        if stats['convergence_warning']:
            print(f"\n⚠️  Discovery rate below threshold - possible convergence")
        
        print("="*60)
    
    def _compute_final_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive final statistics"""
        # Archive data
        archive_df = self.archive.as_pandas(include_solutions=True)
        
        # Discovery rate history
        discovery_history = self.discovery_tracker.get_history()
        discovery_stats = {
            'final_rate': self.discovery_tracker.current_discovery_rate,
            'average_rate': np.mean(discovery_history['discovery_rate']) if discovery_history['discovery_rate'] else 0,
            'convergence_warnings': sum(discovery_history['is_converged']),
            'history': discovery_history
        }
        
        # Adaptive scheduler stats
        scheduler_stats = self.scheduler.get_statistics()
        
        # Visual stealth stats
        stealth_stats = self.archive.get_stats()
        
        # Archive coverage
        max_possible = self.archive.grid_dims[0] * self.archive.grid_dims[1]
        coverage = len(self.archive) / max_possible
        
        return {
            'total_iterations': self.iteration_count,
            'total_evaluations': self.total_evaluations,
            'final_archive_size': len(self.archive),
            'archive_coverage': coverage,
            'best_fitness': archive_df['objective'].max(),
            'mean_fitness': archive_df['objective'].mean(),
            'discovery_rate': discovery_stats,
            'adaptive_scheduler': scheduler_stats,
            'visual_stealth': stealth_stats,
            'archive_data': archive_df
        }
    
    def _print_final_summary(self, stats: Dict[str, Any]):
        """Print comprehensive final summary"""
        print("\n" + "="*60)
        print("🏁 Final Results")
        print("="*60)
        
        print(f"\n📊 Archive Statistics:")
        print(f"  Total Elites: {stats['final_archive_size']}")
        print(f"  Coverage: {stats['archive_coverage']:.2%}")
        print(f"  Best Fitness: {stats['best_fitness']:.4f}")
        print(f"  Mean Fitness: {stats['mean_fitness']:.4f}")
        
        print(f"\n🔍 Discovery Rate:")
        dr = stats['discovery_rate']
        print(f"  Final Rate: {dr['final_rate']:.4f} new/iter")
        print(f"  Average Rate: {dr['average_rate']:.4f} new/iter")
        print(f"  Convergence Warnings: {dr['convergence_warnings']}")
        
        print(f"\n⚙️  Adaptive Scheduler:")
        sched = stats['adaptive_scheduler']
        print(f"  Final Sigma: {sched['final_sigma']:.4f}")
        print(f"  Final Epsilon: {sched['final_epsilon']:.4f}")
        print(f"  Total Adaptations: {sched['total_adaptations']}")
        print(f"  Sigma Resets: {sched['sigma_resets']}")
        
        print(f"\n🎯 Visual Stealth:")
        vs = stats['visual_stealth']
        print(f"  Fitness Improvements: {vs['fitness_improvements']}")
        print(f"  Stealth Improvements: {vs['stealth_improvements']}")
        print(f"  Rejection Rate: {vs['rejection_rate']:.2%}")
        
        print("="*60)
    
    def export_golden_elites(
        self,
        output_dir: Path,
        top_k: int = 5,
        fitness_weight: float = 0.7
    ):
        """
        Export top golden elites with comprehensive visualization
        
        Args:
            output_dir: Output directory for results
            top_k: Number of top elites to export
            fitness_weight: Weight for fitness in scoring (1-weight for BC1)
        """
        if self.verbose:
            print(f"\n🌟 Exporting Top {top_k} Golden Elites...")
        
        export_golden_elites(
            archive=self.archive,
            target_image=self.target_image,
            pert_gen=self.pert_gen,
            vlm_model=self.vlm_model,
            original_caption=self.original_caption,
            groundtruth_caption=self.groundtruth_caption,
            save_dir=str(output_dir),
            num_elites=top_k,
            fitness_weight=fitness_weight,
            generate_captions=True
        )
        
        if self.verbose:
            print(f"✅ Golden elites saved to {output_dir}")
    
    def save_results(self, output_dir: Path):
        """
        Save all results to output directory
        
        Args:
            output_dir: Output directory path
        """
        from src.utils.visualization import plot_heatmap, plot_training_curves, plot_adaptive_sigma_history
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save archive
        archive_df = self.archive.as_pandas(include_solutions=True)
        archive_df.to_csv(output_dir / "archive.csv", index=False)
        
        # Save discovery rate history
        self.discovery_tracker.save_history(output_dir / "discovery_rate.npz")
        
        # Save adaptive scheduler history
        self.scheduler.save_history(output_dir / "adaptive_scheduler.npz")
        
        # Save final statistics
        final_stats = self._compute_final_statistics()
        np.savez(
            output_dir / "final_statistics.npz",
            **{k: v for k, v in final_stats.items() 
               if not isinstance(v, (dict, object))}
        )
        
        # Generate visualizations
        if self.verbose:
            print("\n📊 Generating visualizations...")
        
        # 1. Plot heatmap
        heatmap_path = output_dir / 'final_heatmap.png'
        try:
            plot_heatmap(
                self.archive,  # Pass wrapper - function accesses .archive internally
                save_path=str(heatmap_path),
                title=f'QD Archive Heatmap',
                xlabel='BC1 (L∞ Norm)',
                ylabel='BC2 (Spectral Energy)'
            )
            if self.verbose:
                print(f"  ✓ Saved: final_heatmap.png")
        except Exception as e:
            print(f"  ✗ Failed to generate heatmap: {e}")
        
        # 2. Plot training curves
        curves_path = output_dir / 'training_curves.png'
        try:
            # Get QD metrics history
            history = {
                'qd_score': self.qd_score_history,
                'coverage': self.coverage_history,
                'max_fitness': self.max_fitness_history,
            }
            plot_training_curves(
                history,
                save_path=str(curves_path),
                title='QD Training Progress'
            )
            if self.verbose:
                print(f"  ✓ Saved: training_curves.png")
        except Exception as e:
            print(f"  ✗ Failed to generate training curves: {e}")
        
        # 3. Plot adaptive scheduler history
        adaptive_plot_path = output_dir / 'adaptive_sigma_plot.png'
        try:
            stats = self.scheduler.get_statistics()
            # Build history dict compatible with plot_adaptive_sigma_history
            adaptive_history = {
                'iteration': list(range(1, len(stats['sigma_history']) + 1)),
                'num_elites': [len(self.archive)] * len(stats['sigma_history']),  # Approximate
                'sigma': stats['sigma_history'],
                'epsilon': stats['epsilon_history'],
                'stagnation_counter': [0] * len(stats['sigma_history']),  # Not tracked
                'event': ['none'] * len(stats['sigma_history'])  # Simplified
            }
            plot_adaptive_sigma_history(
                adaptive_history,
                save_path=str(adaptive_plot_path),
                title='Adaptive Sigma History'
            )
            if self.verbose:
                print(f"  ✓ Saved: adaptive_sigma_plot.png")
        except Exception as e:
            print(f"  ✗ Failed to generate adaptive plot: {e}")
        
        if self.verbose:
            print(f"\n💾 Results saved to {output_dir}")
