"""
Visualization utilities for QD results and perturbations
"""
import sys
import os
# Use local pyribs if available
if os.path.exists('/root/ICAT/pyribs') and '/root/ICAT/pyribs' not in sys.path:
    sys.path.insert(0, '/root/ICAT/pyribs')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import os as os_module


def plot_heatmap(archive, 
                save_path: str = None,
                title: str = "MAP-Elites Heatmap",
                xlabel: str = "Behavior Characteristic 1",
                ylabel: str = "Behavior Characteristic 2",
                cmap: str = "viridis"):
    """
    Plot MAP-Elites heatmap
    
    Args:
        archive: QDArchive instance
        save_path: Path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name
    """
    from ribs.visualize import grid_archive_heatmap
    
    plt.figure(figsize=(10, 8))
    grid_archive_heatmap(archive.archive, vmin=None, vmax=None, cmap=cmap)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()


def plot_training_curves(history: Dict[str, List],
                         save_path: str = None,
                         title: str = "Training Progress"):
    """
    Plot training curves for QD metrics
    
    Args:
        history: Dictionary with metric histories
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    metrics = ['qd_score', 'coverage', 'max_fitness', 'mean_fitness']
    titles = ['QD Score', 'Archive Coverage', 'Max Fitness', 'Mean Fitness']
    
    for idx, (metric, metric_title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in history and len(history[metric]) > 0:
            iterations = range(1, len(history[metric]) + 1)
            ax.plot(iterations, history[metric], linewidth=2)
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel(metric_title, fontsize=11)
            ax.set_title(metric_title, fontsize=12)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os_module.makedirs(os_module.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def visualize_perturbations(original_image: np.ndarray,
                           perturbed_images: np.ndarray,
                           perturbations: np.ndarray,
                           save_path: str = None,
                           max_display: int = 5):
    """
    Visualize original, perturbed images, and perturbations
    
    Args:
        original_image: Original image [C, H, W]
        perturbed_images: Perturbed images [N, C, H, W]
        perturbations: Perturbations [N, C, H, W]
        save_path: Path to save figure
        max_display: Maximum number of examples to display
    """
    n_display = min(max_display, perturbed_images.shape[0])
    
    fig, axes = plt.subplots(3, n_display + 1, figsize=(3 * (n_display + 1), 9))
    
    # Helper function to convert image for display
    def prepare_image(img):
        if img.shape[0] == 3:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        return np.clip(img, 0, 1)
    
    # Plot original image
    axes[0, 0].imshow(prepare_image(original_image))
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].text(0.5, 0.5, 'Original\nImage', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    # Plot perturbed images and perturbations
    for i in range(n_display):
        # Perturbed image
        axes[0, i + 1].imshow(prepare_image(perturbed_images[i]))
        axes[0, i + 1].set_title(f'Perturbed {i+1}', fontsize=10)
        axes[0, i + 1].axis('off')
        
        # Perturbation (amplified for visualization)
        pert = perturbations[i]
        pert_vis = prepare_image(pert)
        
        # Amplify perturbation for visibility
        pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        
        axes[1, i + 1].imshow(pert_vis, cmap='seismic', vmin=0, vmax=1)
        axes[1, i + 1].set_title(f'Perturbation {i+1}', fontsize=10)
        axes[1, i + 1].axis('off')
        
        # Difference
        diff = np.abs(perturbed_images[i] - original_image)
        axes[2, i + 1].imshow(prepare_image(diff), cmap='hot')
        axes[2, i + 1].set_title(f'Difference {i+1}', fontsize=10)
        axes[2, i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os_module.makedirs(os_module.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Perturbation visualization saved to {save_path}")
    
    plt.show()


def plot_archive_evolution(archives: List,
                           save_path: str = None,
                           iterations: List[int] = None):
    """
    Plot evolution of archive over iterations
    
    Args:
        archives: List of archive snapshots
        save_path: Path to save figure
        iterations: Iteration numbers for each snapshot
    """
    from ribs.visualize import grid_archive_heatmap
    
    n_snapshots = len(archives)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5))
    
    if n_snapshots == 1:
        axes = [axes]
    
    for idx, archive in enumerate(archives):
        plt.sca(axes[idx])
        grid_archive_heatmap(archive.archive if hasattr(archive, 'archive') else archive)
        
        if iterations is not None:
            axes[idx].set_title(f'Iteration {iterations[idx]}', fontsize=12)
        else:
            axes[idx].set_title(f'Snapshot {idx+1}', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os_module.makedirs(os_module.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Archive evolution saved to {save_path}")
    
    plt.show()


def save_results_summary(archive, history: Dict, save_dir: str):
    """
    Save comprehensive results summary
    
    Args:
        archive: QDArchive instance
        history: Training history
        save_dir: Directory to save results
    """
    os_module.makedirs(save_dir, exist_ok=True)
    
    # Save heatmap
    plot_heatmap(
        archive,
        save_path=os_module.path.join(save_dir, 'heatmap.png'),
        title='Final MAP-Elites Archive'
    )
    
    # Save training curves
    plot_training_curves(
        history,
        save_path=os_module.path.join(save_dir, 'training_curves.png'),
        title='Training Progress'
    )
    
    # Save statistics as text
    stats = archive.get_stats()
    with open(os_module.path.join(save_dir, 'final_stats.txt'), 'w') as f:
        f.write("Final Archive Statistics\n")
        f.write("=" * 50 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results summary saved to {save_dir}")


def plot_adaptive_sigma_history(history: Dict[str, List],
                                 save_path: str = None,
                                 title: str = "Adaptive Attack History"):
    """
    Plot adaptive scheduler history (works with both old and new scheduler)
    
    Args:
        history: Dictionary with keys: iteration, num_elites, sigma, epsilon, 
                stagnation_counter, and either 'is_boosted' (old) or 'event' (new)
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    iterations = history['iteration']
    
    # Plot 1: Number of Elites
    axes[0].plot(iterations, history['num_elites'], linewidth=2, color='blue')
    axes[0].set_ylabel('Number of Elites', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Highlight boosted/active periods (support both old and new format)
    if 'event' in history:
        # New format: use 'event' key ('boost', 'reset', 'safety_catch', 'none')
        events = np.array(history['event'])
        boost_indices = np.where((events == 'boost') | (events == 'safety_catch'))[0]
        
        # Highlight individual boost events
        for idx in boost_indices:
            if idx < len(iterations):
                axes[0].axvline(x=iterations[idx], color='red', alpha=0.3, linewidth=1)
        
        # Mark reset events
        reset_indices = np.where(events == 'reset')[0]
        for idx in reset_indices:
            if idx < len(iterations):
                axes[0].axvline(x=iterations[idx], color='green', alpha=0.3, linewidth=1)
        
        if len(boost_indices) > 0 or len(reset_indices) > 0:
            from matplotlib.lines import Line2D
            legend_elements = []
            if len(boost_indices) > 0:
                legend_elements.append(Line2D([0], [0], color='red', alpha=0.3, linewidth=2, label='Boost/Safety'))
            if len(reset_indices) > 0:
                legend_elements.append(Line2D([0], [0], color='green', alpha=0.3, linewidth=2, label='Reset'))
            if legend_elements:
                axes[0].legend(handles=legend_elements)
    
    elif 'is_boosted' in history:
        # Old format: use 'is_boosted' boolean array
        is_boosted = np.array(history['is_boosted'])
        if np.any(is_boosted):
            boosted_indices = np.where(is_boosted)[0]
            if len(boosted_indices) > 0:
                axes[0].axvspan(iterations[boosted_indices[0]], iterations[boosted_indices[-1]], 
                               alpha=0.2, color='red', label='Boosted Period')
                axes[0].legend()
    
    # Plot 2: Sigma
    axes[1].plot(iterations, history['sigma'], linewidth=2, color='green', marker='o', markersize=3)
    axes[1].set_ylabel('Sigma (σ)', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon
    axes[2].plot(iterations, history['epsilon'], linewidth=2, color='orange', marker='s', markersize=3)
    axes[2].set_ylabel('Epsilon (ε)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Stagnation Counter
    axes[3].plot(iterations, history['stagnation_counter'], linewidth=2, color='red', drawstyle='steps-post')
    axes[3].axhline(y=15, color='black', linestyle='--', linewidth=1, label='Stagnation Threshold')
    axes[3].set_ylabel('Stagnation Counter', fontsize=12)
    axes[3].set_xlabel('Iteration', fontsize=12)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os_module.makedirs(os_module.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Adaptive attack history plot saved to {save_path}")
    
    plt.close()
