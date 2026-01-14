"""
Example script: Quick start demo for QD attacks
"""
import numpy as np
import torch
from PIL import Image

from src.models import SigLIPWrapper
from src.qd_engine import QDArchive, QDScheduler, create_emitters
from src.attack import PerturbationGenerator, FitnessFunction, MeasureFunction
from src.utils import plot_heatmap


def quick_demo():
    """Run a quick demo of QD attack"""
    
    print("Quick Demo: QD Black-box Attack on VLM")
    print("=" * 50)
    
    # 1. Load model
    print("\n1. Loading SigLIP model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vlm_model = SigLIPWrapper(
        model_name='ViT-B-16-SigLIP',
        pretrained='webli',
        device=device
    )
    
    # 2. Create a dummy image
    print("\n2. Creating dummy image...")
    image_size = 128
    original_image = np.random.rand(3, image_size, image_size).astype(np.float32)
    original_caption = "A photo of a cat"
    
    # 3. Setup QD components
    print("\n3. Setting up QD components...")
    image_shape = original_image.shape
    solution_dim = np.prod(image_shape)
    
    pert_gen = PerturbationGenerator(image_shape, epsilon=0.05)
    fitness_fn = FitnessFunction(metric='clip_similarity', device=device)
    measure_fn = MeasureFunction(image_shape, measure_types=('linf_norm', 'spectral_energy'))
    
    # Create archive
    archive = QDArchive(
        solution_dim=solution_dim,
        grid_dims=(25, 25),
        ranges=[(0.0, 0.05), (0.0, 1.0)]
    )
    
    # Create emitters
    emitters = create_emitters(
        archive.archive,
        algorithm='cma_me',
        num_emitters=3,
        sigma0=0.01,
        batch_size=15
    )
    
    # Create scheduler
    scheduler = QDScheduler(archive, emitters)
    
    # 4. Run attack for 50 iterations
    print("\n4. Running QD attack (50 iterations)...")
    for iteration in range(1, 51):
        solutions = scheduler.ask()
        perturbations = pert_gen.solutions_to_perturbations(solutions)
        perturbed_images = pert_gen.apply_perturbation(original_image, perturbations)
        
        objectives = fitness_fn.compute_fitness(
            [original_caption] * len(perturbed_images),
            perturbed_images,
            vlm_model
        )
        
        measures = measure_fn.compute_measures(perturbations)
        scheduler.tell(objectives, measures)
        
        if iteration % 10 == 0:
            stats = archive.get_stats()
            print(f"  Iter {iteration}: QD Score = {stats['qd_score']:.2f}, "
                  f"Coverage = {stats['coverage']:.2f}, "
                  f"Max Fitness = {stats['max_fitness']:.3f}")
    
    # 5. Visualize results
    print("\n5. Visualizing results...")
    plot_heatmap(
        archive,
        save_path='/root/ICAT/results/demo_heatmap.png',
        title='Quick Demo: MAP-Elites Heatmap'
    )
    
    final_stats = archive.get_stats()
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\nDemo completed! Check /root/ICAT/results/demo_heatmap.png")


if __name__ == '__main__':
    quick_demo()
