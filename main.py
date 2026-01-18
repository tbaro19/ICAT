"""
Main entry point for Quality-Diversity Black-box Attacks on VLMs
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Import project modules
from src.models.blip2_wrapper import BLIP2Wrapper
from src.models.paligemma_wrapper import PaliGemmaWrapper
from src.models.qwen2vl_wrapper import Qwen2VLWrapper
from src.qd_engine import QDArchive, QDScheduler, create_emitters
from src.qd_engine.adaptive_attack_scheduler import AdaptiveAttackScheduler
from src.qd_engine.visual_stealth_archive import VisualStealthArchive
from src.qd_engine.discovery_tracker import DiscoveryRateTracker
from src.qd_engine.unified_attack_manager import UnifiedAttackManager
from src.attack import PerturbationGenerator, MeasureFunction, JailbreakLogitFitness
from src.utils import plot_heatmap, plot_training_curves, visualize_perturbations
from src.utils import DatasetLoader
from src.utils import export_golden_elites
from src.utils.comparison_viz import create_attack_comparison, compute_image_text_score, generate_caption_from_image
from src.utils.translation import translate_vi_to_en, translate_en_to_vi
from bert_score import score as bert_score


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Quality-Diversity Black-box Attacks on Vision-Language Models'
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='blip2', 
                       choices=['blip2', 'paligemma', 'qwen2vl'],
                       help='VLM model to attack')
    parser.add_argument('--model_name', type=str, default='Salesforce/blip2-opt-2.7b',
                       help='Specific model architecture (HuggingFace model ID)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Pretrained weights (not used for HF models)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ktvic',
                       choices=['coco', 'ktvic', 'uit-viic', 'flickr30k'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/root/ICAT/data',
                       help='Root directory for datasets')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to evaluate')
    
    # Attack arguments
    parser.add_argument('--epsilon', type=float, default=0.12,
                       help='Maximum perturbation magnitude (L-inf)')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Perturbation resolution (will be upsampled to match input)')
    parser.add_argument('--target_size', type=int, default=384,
                       help='Target image size for model evaluation')
    
    # QD algorithm arguments
    parser.add_argument('--algorithm', type=str, default='cma_me',
                       choices=['map_elites', 'cma_me', 'cma_mae', 'cma_mega'],
                       help='QD algorithm to use')
    parser.add_argument('--num_emitters', type=int, default=2,
                       help='Number of emitters')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of QD iterations')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--sigma0', type=float, default=0.02,
                       help='Initial step size for evolution strategies')
    
    # Archive arguments
    parser.add_argument('--grid_dims', type=int, nargs=2, default=[10, 10],
                       help='Dimensions of MAP-Elites grid (10x10 for fine-grained jailbreak diversity)')
    parser.add_argument('--bc_types', type=str, nargs=2, 
                       default=['linf_norm', 'spectral_energy'],
                       help='Types of behavior characteristics')
    parser.add_argument('--bc_ranges', type=float, nargs=4,
                       default=[0.05, 0.10, 0.10, 0.18],
                       help='🔒 Safety Red-Team Ranges: L-inf [0.05, 0.10], Spectral [0.10, 0.18] for stealthy jailbreak triggers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='/root/ICAT/results',
                       help='Directory to save results')
    parser.add_argument('--exp_name', type=str, default='experiment',
                       help='Experiment name')
    parser.add_argument('--save_interval', type=int, default=0,
                       help='Save interval for checkpoints (0=auto: max(2, iterations//10))')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Unified framework arguments
    parser.add_argument('--use_unified', action='store_true',
                       help='Use Unified Adaptive Adversarial Framework')
    parser.add_argument('--initial_epsilon', type=float, default=0.07,
                       help='Initial epsilon for adaptive scheduler')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create organized output directory: results/{algo_name}/{model_architecture}/{dataset}/{exp_name}/
    # Extract clean model name (remove special chars and simplify)
    model_arch = args.model_name.replace('-', '_').replace('.', '_')
    model_folder = f"{args.model}_{model_arch}"
    
    exp_dir = os.path.join(
        args.output_dir,
        args.algorithm,
        model_folder,
        args.dataset,
        args.exp_name
    )
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Results will be saved to: {exp_dir}")
    
    # Auto-adjust save_interval based on iterations
    if args.save_interval == 0:
        # Save every 100 iterations (or fewer for short runs)
        args.save_interval = min(100, max(10, args.iterations // 5))
    print(f"Save interval: every {args.save_interval} iterations")
    
    # ========================================
    # 1. Load VLM Model
    # ========================================
    print("\n" + "="*50)
    print("Loading Vision-Language Model")
    print("="*50)
    
    if args.model == 'blip2':
        vlm_model = BLIP2Wrapper(
            model_name=args.model_name,
            device=device
        )
    elif args.model == 'paligemma':
        vlm_model = PaliGemmaWrapper(
            model_name=args.model_name,
            device=device
        )
    elif args.model == 'qwen2vl':
        vlm_model = Qwen2VLWrapper(
            model_name=args.model_name,
            device=device
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # ========================================
    # 2. Load Dataset
    # ========================================
    print("\n" + "="*50)
    print("Loading Dataset")
    print("="*50)
    
    loader = DatasetLoader(data_root=args.data_root)
    
    # Don't limit samples when we need to find specific images
    use_max_samples = None  # Load full dataset to find specific images
    
    try:
        if args.dataset == 'coco':
            dataset = loader.load_coco(
                split='val',
                transform=None,
                max_samples=use_max_samples
            )
        elif args.dataset == 'ktvic':
            dataset = loader.load_ktvic(
                split='test',
                transform=None,
                max_samples=use_max_samples
            )
        elif args.dataset == 'uit-viic':
            dataset = loader.load_uit_viic(
                split='test',
                transform=None,
                max_samples=use_max_samples
            )
        elif args.dataset == 'flickr30k':
            dataset = loader.load_flickr30k(
                transform=None,
                max_samples=use_max_samples
            )
        else:
            raise ValueError(f"Dataset {args.dataset} not implemented yet")
        
        print(f"Loaded {len(dataset)} samples from {args.dataset}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Creating dummy dataset for demonstration...")
        
        # Create dummy data
        dummy_images = torch.randn(10, 3, args.image_size, args.image_size)
        dummy_captions = [f"A sample caption {i}" for i in range(10)]
        dummy_files = [f"image_{i}.jpg" for i in range(10)]
        
        from src.utils.data_loader import ImageCaptionDataset
        dataset = ImageCaptionDataset(
            images=dummy_images,
            captions=dummy_captions,
            image_files=dummy_files,
            transform=None
        )
    
    # Define specific images to use - DIRECT PATHS
    specific_image_paths = {
        'flickr30k': '/root/ICAT/data/Flickr30k/flickr30k_images/flickr30k_images/flickr30k_images/4985704.jpg',
        'ktvic': '/root/ICAT/data/KTVIC/ktvic_dataset/public-test-images/00000000001.jpg',
        'uit-viic': '/root/ICAT/data/UIT-ViIC/uitvic_dataset/coco_uitvic_test/coco_uitvic_test/000000048604.jpg'
    }
    
    # Try to load the specific image directly
    sample_idx = 0
    use_specific_image = False
    specific_image_data = None
    
    if args.dataset in specific_image_paths:
        specific_path = specific_image_paths[args.dataset]
        print(f"\n{'='*60}")
        print(f"LOADING SPECIFIC IMAGE: {os.path.basename(specific_path)}")
        print(f"Path: {specific_path}")
        print(f"{'='*60}")
        
        if os.path.exists(specific_path):
            try:
                from PIL import Image
                import torchvision.transforms as T
                
                # Load image directly
                img_pil = Image.open(specific_path).convert('RGB')
                
                # Convert to tensor [C, H, W]
                transform = T.ToTensor()
                specific_image_data = transform(img_pil).numpy()
                
                # Find corresponding caption in dataset by searching for this image
                target_filename = os.path.basename(specific_path)
                print(f"    Searching for caption matching: {target_filename}")
                
                found_caption = False
                for idx in range(len(dataset)):
                    try:
                        # Check image_files list (this contains just filenames)
                        if dataset.image_files[idx] == target_filename:
                            sample_idx = idx
                            found_caption = True
                            print(f"✓ Found caption at index {idx}")
                            print(f"  Caption: {dataset.captions[idx][:100]}...")
                            break
                    except Exception as e:
                        print(f"Error checking index {idx}: {e}")
                        continue
                
                if not found_caption:
                    print(f"✗ Caption not found for {target_filename}")
                    print(f"  First few filenames in dataset:")
                    for i in range(min(5, len(dataset))):
                        try:
                            print(f"    [{i}] {dataset.image_files[i]}")
                        except Exception as e:
                            print(f"    [{i}] Error: {e}")
                
                use_specific_image = True
                print(f"✓✓✓ SUCCESSFULLY LOADED SPECIFIC IMAGE")
                print(f"    Shape: {specific_image_data.shape}")
            except Exception as e:
                print(f"✗ Failed to load specific image: {e}")
                print(f"    Falling back to dataset index 0")
        else:
            print(f"✗ Specific image not found at path")
            print(f"    Falling back to dataset index 0")
    
    # Get caption from dataset
    if use_specific_image:
        original_image = specific_image_data
        original_caption = dataset.captions[sample_idx]  # Get caption directly from list
        print(f"\n✓✓✓ Using caption from index {sample_idx}: {original_caption[:100]}...")
    else:
        sample_idx = 0
        original_image, original_caption, _ = dataset[sample_idx]
    
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.numpy()
    
    # Translate Vietnamese caption to English immediately
    if args.dataset in ['ktvic', 'uit-viic']:
        print(f"\nOriginal caption (Vietnamese): {original_caption}")
        original_caption = translate_vi_to_en(original_caption)
        print(f"Translated caption (English): {original_caption}")
    
    print(f"\nUsing caption for experiment: {original_caption}")
    
    # Save groundtruth caption
    groundtruth_caption = original_caption
    
    # Save high-res image for visualization before resizing
    original_image_highres = original_image.copy()
    print(f"High-res image saved for visualization: {original_image_highres.shape}")
    
    # Resize target image to target_size for model evaluation
    target_size = args.target_size
    if original_image.shape[1] != target_size or original_image.shape[2] != target_size:
        import torch.nn.functional as F
        img_tensor = torch.from_numpy(original_image).unsqueeze(0)
        target_image_tensor = F.interpolate(img_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
        target_image = target_image_tensor.squeeze(0).numpy()
    else:
        target_image = original_image
    
    # Generate caption for clean image
    print("\nGenerating caption for clean image...")
    try:
        clean_image_for_caption = torch.from_numpy(target_image).float()
        original_caption_generated = generate_caption_from_image(vlm_model, clean_image_for_caption)
        print(f"Generated caption (clean): {original_caption_generated}")
        print(f"Groundtruth caption: {groundtruth_caption}")
    except Exception as e:
        print(f"Warning: Error generating caption for clean image: {e}")
        original_caption_generated = groundtruth_caption
    
    # Resize to low-res for perturbation optimization
    if original_image.shape[1] != args.image_size or original_image.shape[2] != args.image_size:
        img_tensor = torch.from_numpy(original_image).unsqueeze(0)
        img_resized = F.interpolate(img_tensor, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
        original_image = img_resized.squeeze(0).numpy()
        print(f"Low-res perturbation search: {args.image_size}x{args.image_size} → upsampled to {target_size}x{target_size}")
    
    # Resize for faster computation if needed
    if original_image.shape[-1] != args.image_size:
        from PIL import Image
        import torchvision.transforms as T
        
        resize_transform = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor()
        ])
        
        # Convert to PIL and resize
        img_pil = Image.fromarray((original_image.transpose(1, 2, 0) * 255).astype(np.uint8))
        original_image = resize_transform(img_pil).numpy()
    
    print(f"Perturbation search shape: {original_image.shape}")
    print(f"Target evaluation size: {target_size}x{target_size}")
    print(f"Original caption: {original_caption}")
    
    # ========================================
    # 3. Initialize QD Components
    # ========================================
    print("\n" + "="*50)
    print("Initializing QD Components")
    print("="*50)
    
    image_shape = original_image.shape  # (C, H, W) - low res
    solution_dim = np.prod(image_shape)
    
    print(f"Solution dimension: {solution_dim:,} (CMA-ES covariance: ~{solution_dim**2*8/1e6:.1f} MB)")
    
    # Initialize perturbation generator with upsampling
    pert_gen = PerturbationGenerator(
        image_shape=image_shape,
        epsilon=args.epsilon,
        target_size=target_size  # Upsample to this size before applying
    )
    
    # Initialize jailbreak fitness function
    print("\n🎯 SAFETY RED-TEAMING MODE - Jailbreak Research")
    print("="*70)
    print("Fitness: Harmful Token Lexicon (Targeted Logit-Forcing)")
    print("Objective: Maximize log-probability of prohibited tokens")
    print("BC1: L∞ Stealthiness [0.05, 0.10] | BC2: Spectral [0.10, 0.18]")
    print("="*70)
    
    fitness_fn = JailbreakLogitFitness(
        model_name=model_name,
        device=device,
        chunk_size=1,  # VLMs are memory-intensive
        verbose=True
    )
    fitness_fn.lexicon.print_lexicon_summary()
    
    # Initialize measure function
    measure_fn = MeasureFunction(
        image_shape=image_shape,
        measure_types=tuple(args.bc_types)
    )
    
    # Create Visual Stealth Archive (prioritizes minimal perturbation)
    bc_ranges = [
        (args.bc_ranges[0], args.bc_ranges[1]),
        (args.bc_ranges[2], args.bc_ranges[3])
    ]
    
    print("\n🔒 BC Ranges LOCKED at: [{:.2f}, {:.2f}] x [{:.2f}, {:.2f}]".format(*args.bc_ranges))
    print("   This ensures consistent measurement across all 1000 iterations.")
    
    archive = VisualStealthArchive(
        solution_dim=solution_dim,
        grid_dims=tuple(args.grid_dims),
        ranges=bc_ranges,
        learning_rate=1.0,
        threshold_min=-np.inf
    )
    
    # Create emitters
    emitters = create_emitters(
        archive=archive.archive,
        algorithm=args.algorithm,
        num_emitters=args.num_emitters,
        sigma0=args.sigma0,
        x0=np.zeros(solution_dim),
        batch_size=args.batch_size
    )
    
    # Create scheduler
    scheduler = QDScheduler(archive, emitters)
    
    # Create adaptive attack scheduler (epsilon creep + sigma ramping)
    adaptive_scheduler = AdaptiveAttackScheduler(
        emitters=emitters,
        perturbation_generator=pert_gen,
        baseline_sigma=args.sigma0,
        baseline_epsilon=args.epsilon,
        stagnation_threshold=15,
        sigma_multiplier=1.5,       # 1.5x increase (not 2x)
        epsilon_increment=0.01,     # +0.01 per boost
        epsilon_max=0.20,           # Max epsilon limit
        sigma_max=0.10,             # Safety catch at 0.10
        sigma_reset=0.05,           # Reset value when exceeding max
        epsilon_decrease=0.005      # Slight decrease on progress
    )
    
    # Create Discovery Rate Tracker (convergence detection)
    discovery_tracker = DiscoveryRateTracker(
        window_size=50,              # Rolling window of 50 iterations
        convergence_threshold=0.02,  # Less than 0.02 new elites/iter
        consecutive_warnings=2       # Warn after 2 consecutive low-rate windows
    )
    
    print(f"QD Archive: {args.grid_dims[0]}x{args.grid_dims[1]} grid (Visual Stealth Mode)")
    print(f"  Strategy: Efficiency over Brutality (prioritize low L∞ perturbation)")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Solution dimension: {solution_dim}")
    print(f"Adaptive Attack Scheduler: Enabled")
    print(f"  Stagnation threshold: 15 iterations")
    print(f"  Epsilon: {args.epsilon:.2f} → {0.20:.2f} (max)")
    print(f"  Sigma: {args.sigma0:.2f} → {0.10:.2f} (max, resets to 0.05)")
    print(f"Discovery Rate Tracker: Enabled")
    print(f"  Window size: 50 iterations")
    print(f"  Convergence threshold: 0.02 new elites/iter")
    
    # ========================================
    # 4. Run QD Attack Loop
    # ========================================
    print("\n" + "="*50)
    
    # Check if using unified framework
    if args.use_unified:
        print("🚀 Using Unified Adaptive Adversarial Framework")
        print("="*50)
        
        # Create unified attack manager
        attack_manager = UnifiedAttackManager(
            vlm_model=vlm_model,
            archive=archive,
            base_scheduler=scheduler,
            pert_gen=pert_gen,
            measure_fn=measure_fn,
            target_image=target_image,
            original_caption=original_caption,
            groundtruth_caption=groundtruth_caption,
            bc_ranges=bc_ranges,
            grid_dims=tuple(args.grid_dims),
            use_logit_loss=args.use_logit_loss,
            initial_epsilon=args.initial_epsilon,
            device=device,
            chunk_size=args.batch_size,
            verbose=True
        )
        
        # Run unified attack
        final_stats = attack_manager.run(
            num_iterations=args.iterations,
            checkpoint_interval=args.save_interval
        )
        
        # Save all results
        attack_manager.save_results(exp_dir)
        
        # Export golden elites
        attack_manager.export_golden_elites(
            output_dir=exp_dir,
            top_k=5,
            fitness_weight=0.7
        )
        
        print("\n✅ Unified framework execution complete!")
        return
    
    # Original execution path (if not using unified framework)
    print("Running QD Attack Loop")
    print("="*50)
    
    # Cache groundtruth caption embedding for faster evaluation
    with torch.no_grad():
        groundtruth_text_emb = vlm_model.get_text_embeddings([groundtruth_caption])
        original_text_emb = vlm_model.get_text_embeddings([original_caption_generated])
    
    for iteration in tqdm(range(1, args.iterations + 1), desc="QD Iterations"):
        # 1. Ask for new solutions
        solutions = scheduler.ask()
        
        # Process solutions in smaller chunks to avoid OOM
        chunk_size = min(4, len(solutions))
        num_chunks = (len(solutions) + chunk_size - 1) // chunk_size
        
        all_objectives = []
        all_measures = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(solutions))
            chunk_solutions = solutions[start_idx:end_idx]
            
            # 2. Convert solutions to perturbations (upsampled to target_size internally)
            perturbations = pert_gen.solutions_to_perturbations(chunk_solutions)
            
            # 3. Apply perturbations to target high-res image
            perturbed_images = pert_gen.apply_perturbation(target_image, perturbations)
            
            # 4. Evaluate fitness (attack effectiveness)
            objectives = fitness_fn.compute_fitness(
                original_captions=[original_caption] * len(perturbed_images),
                perturbed_images=perturbed_images,
                vlm_model=vlm_model,
                original_embeddings=original_text_emb
            )
            
            # 5. Compute behavior characteristics
            measures = measure_fn.compute_measures(perturbations)
            
            all_objectives.extend(objectives)
            all_measures.append(measures)
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine all chunks
        all_measures = np.vstack(all_measures)
        
        # 6. Tell scheduler the results (Visual Stealth Archive handles replacement logic)
        scheduler.tell(np.array(all_objectives), all_measures)
        
        # 7. Update adaptive attack scheduler
        stats = archive.get_stats()
        adaptive_status = adaptive_scheduler.step(iteration, stats['num_elites'])
        
        # 8. Update discovery rate tracker
        discovery_status = discovery_tracker.update(stats['num_elites'])
        
        # Print convergence warning if triggered
        if discovery_status['message'] is not None:
            print(discovery_status['message'])
        
        # 9. Save checkpoints and log statistics
        if iteration % args.save_interval == 0:
            print(f"\n[{iteration}/{args.iterations}] Checkpoint:")
            print(f"  QD Score: {stats['qd_score']:.4f}")
            print(f"  Coverage: {stats['coverage']:.4f}")
            print(f"  Max Fitness: {stats['max_fitness']:.4f}")
            print(f"  Num Elites: {stats['num_elites']}")
            print(f"  {discovery_tracker.get_status_string()}")
            print(f"  Adaptive Attack: σ={adaptive_status['sigma']:.4f}, ε={adaptive_status['epsilon']:.4f}, stagnation={adaptive_status['stagnation_counter']}")
            
            # Log Visual Stealth Archive statistics
            if iteration % (args.save_interval * 2) == 0:  # Every other checkpoint
                print(f"\n  {archive.get_replacement_summary()}")
            
            # Log BC value statistics
            if len(archive.archive) > 0:
                all_elites = list(archive.archive)
                bc1_values = [elite['measures'][0] for elite in all_elites]
                bc2_values = [elite['measures'][1] for elite in all_elites]
                print(f"  BC1 ({args.bc_types[0]}): min={min(bc1_values):.4f}, max={max(bc1_values):.4f}, mean={np.mean(bc1_values):.4f}")
                print(f"  BC2 ({args.bc_types[1]}): min={min(bc2_values):.4f}, max={max(bc2_values):.4f}, mean={np.mean(bc2_values):.4f}")
            
            # Save archive
            checkpoint_file = os.path.join(exp_dir, f'archive_iter_{iteration}.pkl')
            archive.save(checkpoint_file)
            print(f"  ✓ Saved: {checkpoint_file}")
    
    # ========================================
    # 5. Visualize and Save Results
    # ========================================
    print("\n" + "="*50)
    print("Saving Final Results")
    print("="*50)
    
    saved_files = []
    
    # Save adaptive attack scheduler history
    adaptive_history_path = os.path.join(exp_dir, 'adaptive_attack_history.pkl')
    adaptive_scheduler.save_history(adaptive_history_path)
    saved_files.append(adaptive_history_path)
    
    # Save discovery tracker history
    discovery_history_path = os.path.join(exp_dir, 'discovery_tracker_history.pkl')
    discovery_tracker.save_history(discovery_history_path)
    saved_files.append(discovery_history_path)
    print(f"✓ Saved: discovery_tracker_history.pkl")
    
    # Plot heatmap
    heatmap_path = os.path.join(exp_dir, 'final_heatmap.png')
    plot_heatmap(
        archive,
        save_path=heatmap_path,
        title=f'MAP-Elites: {args.model.upper()} on {args.dataset.upper()}',
        xlabel=args.bc_types[0],
        ylabel=args.bc_types[1]
    )
    saved_files.append(heatmap_path)
    print(f"✓ Saved: final_heatmap.png")
    
    # Plot training curves
    history = scheduler.get_history()
    curves_path = os.path.join(exp_dir, 'training_curves.png')
    plot_training_curves(
        history,
        save_path=curves_path,
        title='QD Training Progress'
    )
    saved_files.append(curves_path)
    print(f"✓ Saved: training_curves.png")
    
    # Plot adaptive attack history
    from src.utils import plot_adaptive_sigma_history
    adaptive_history = adaptive_scheduler.get_history()
    adaptive_plot_path = os.path.join(exp_dir, 'adaptive_attack_plot.png')
    plot_adaptive_sigma_history(
        adaptive_history,
        save_path=adaptive_plot_path,
        title=f'Adaptive Attack History - {args.model.upper()} on {args.dataset.upper()}'
    )
    saved_files.append(adaptive_plot_path)
    print(f"✓ Saved: adaptive_attack_plot.png")
    
    # Visualize best perturbations
    best_elite = archive.get_best_elite()
    if best_elite is not None:
        best_solution = best_elite['solution']
        best_pert = pert_gen.solutions_to_perturbations(best_solution[np.newaxis, :])
        best_perturbed = pert_gen.apply_perturbation(target_image, best_pert)
        
        pert_path = os.path.join(exp_dir, 'best_perturbations.png')
        visualize_perturbations(
            target_image,  # Use high-res target for visualization
            best_perturbed,
            best_pert,
            save_path=pert_path,
            max_display=1
        )
        saved_files.append(pert_path)
        print(f"✓ Saved: best_perturbations.png")
        
        # Export Golden Elites (Top 5 with best fitness/stealth balance)
        print("\n" + "="*60)
        print("🏆 EXPORTING GOLDEN ELITES")
        print("="*60)
        
        golden_files = export_golden_elites(
            archive=archive,
            pert_gen=pert_gen,
            target_image=original_image_highres,
            original_caption=original_caption,
            groundtruth_caption=groundtruth_caption,
            vlm_model=vlm_model,
            save_dir=exp_dir,
            num_elites=5,
            fitness_weight=0.7,
            bc1_weight=0.3,
            generate_captions=True
        )
        saved_files.extend(golden_files)
        
        print(f"\nBest Elite:")
        print(f"  Fitness: {best_elite['objective']:.4f}")
        print(f"  BC1 (L∞ Norm): {best_elite['measures'][0]:.4f}")
        print(f"  BC2 (Spectral): {best_elite['measures'][1]:.4f}")
    
    # Save final statistics
    final_stats = archive.get_stats()
    
    # Compute BC statistics from all elites
    bc_stats = {}
    if len(archive.archive) > 0:
        all_elites = list(archive.archive)
        bc1_values = [elite['measures'][0] for elite in all_elites]
        bc2_values = [elite['measures'][1] for elite in all_elites]
        bc_stats = {
            'bc1_min': min(bc1_values),
            'bc1_max': max(bc1_values),
            'bc1_mean': np.mean(bc1_values),
            'bc2_min': min(bc2_values),
            'bc2_max': max(bc2_values),
            'bc2_mean': np.mean(bc2_values)
        }
    
    stats_path = os.path.join(exp_dir, 'final_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("Final QD Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Algorithm: {args.algorithm.upper()}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write("\n")
        for key, value in final_stats.items():
            f.write(f"{key}: {value}\n")
        
        # Add Visual Stealth Archive statistics
        f.write("\n" + "=" * 50 + "\n")
        f.write("Visual Stealth Archive (Niche Optimization)\n")
        f.write("=" * 50 + "\n")
        f.write(archive.get_replacement_summary())
        
        # Add Discovery Rate Tracker summary
        discovery_history = discovery_tracker.get_history()
        if len(discovery_history['discovery_rate']) > 0:
            final_discovery_rate = discovery_history['discovery_rate'][-1]
            avg_discovery_rate = np.mean([r for r in discovery_history['discovery_rate'] if r > 0])
            f.write("\n" + "=" * 50 + "\n")
            f.write("Discovery Rate Tracking\n")
            f.write("=" * 50 + "\n")
            f.write(f"Final Discovery Rate: {final_discovery_rate:.4f} new elites/iter\n")
            f.write(f"Average Discovery Rate: {avg_discovery_rate:.4f} new elites/iter\n")
            f.write(f"Convergence Warnings: {sum(discovery_history['is_converged'])}\n")
        
        # Add BC statistics
        if bc_stats:
            f.write("\n" + "=" * 50 + "\n")
            f.write("Behavior Characteristic Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"BC1 ({args.bc_types[0]}): [{bc_stats['bc1_min']:.4f}, {bc_stats['bc1_max']:.4f}], mean={bc_stats['bc1_mean']:.4f}\n")
            f.write(f"BC2 ({args.bc_types[1]}): [{bc_stats['bc2_min']:.4f}, {bc_stats['bc2_max']:.4f}], mean={bc_stats['bc2_mean']:.4f}\n")
            f.write(f"\n🔒 Configured Ranges (LOCKED): BC1=[{args.bc_ranges[0]}, {args.bc_ranges[1]}], BC2=[{args.bc_ranges[2]}, {args.bc_ranges[3]}]\n")
    saved_files.append(stats_path)
    print(f"✓ Saved: final_stats.txt")
    
    # Save final archive
    final_archive_path = os.path.join(exp_dir, 'archive_final.pkl')
    archive.save(final_archive_path)
    saved_files.append(final_archive_path)
    print(f"✓ Saved: archive_final.pkl")
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Results directory: {exp_dir}")
    print(f"\nSaved {len(saved_files)} files:")
    for f in saved_files:
        print(f"  • {os.path.basename(f)}")
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Jailbreak safety red-teaming metrics
    if hasattr(fitness_fn, 'compute_jsr'):
        print("\n" + "="*70)
        print("🎯 JAILBREAK SAFETY RED-TEAMING METRICS")
        print("="*70)
        
        # Compute JSR (Jailbreak Success Rate)
        jsr = fitness_fn.compute_jsr(archive)
        print(f"📊 Jailbreak Success Rate (JSR): {jsr:.2f}%")
        print(f"   (Percentage of archive bins with harmful token activation)")
        
        # Find Infiltration Elites (most stealthy jailbreaks)
        infiltration_elites = fitness_fn.find_infiltration_elites(archive, top_k=5)
        if len(infiltration_elites) > 0:
            print(f"\n🔍 Infiltration Depth (Top {len(infiltration_elites)} Stealthiest Jailbreaks):")
            for i, elite in enumerate(infiltration_elites, 1):
                print(f"   #{i}: L-inf={elite['bc1_linf']:.4f}, Fitness={elite['objective']:.4f}, Spectral={elite['bc2_spectral']:.4f}")
            print(f"\n   🏆 Best Infiltration: L-inf = {infiltration_elites[0]['bc1_linf']:.4f}")
        else:
            print("\n⚠️  No successful jailbreaks found in archive")
        
        # Print fitness summary
        fitness_fn.print_summary()
        print("="*70)
    
    # Print Visual Stealth Archive statistics
    print("\n" + archive.get_replacement_summary())
    
    # Print Discovery Rate summary
    discovery_history = discovery_tracker.get_history()
    if len(discovery_history['discovery_rate']) > 0:
        final_discovery_rate = discovery_history['discovery_rate'][-1]
        avg_discovery_rate = np.mean([r for r in discovery_history['discovery_rate'] if r > 0])
        print("\nDiscovery Rate Summary:")
        print(f"  Final: {final_discovery_rate:.4f} new elites/iter")
        print(f"  Average: {avg_discovery_rate:.4f} new elites/iter")
        print(f"  Convergence Warnings: {sum(discovery_history['is_converged'])}")
    
    # Print BC statistics
    if bc_stats:
        print("\nBehavior Characteristic Statistics:")
        print(f"  BC1 ({args.bc_types[0]}): [{bc_stats['bc1_min']:.4f}, {bc_stats['bc1_max']:.4f}], mean={bc_stats['bc1_mean']:.4f}")
        print(f"  BC2 ({args.bc_types[1]}): [{bc_stats['bc2_min']:.4f}, {bc_stats['bc2_max']:.4f}], mean={bc_stats['bc2_mean']:.4f}")
        print(f"\n🔒 Configured Ranges (LOCKED): BC1=[{args.bc_ranges[0]}, {args.bc_ranges[1]}], BC2=[{args.bc_ranges[2]}, {args.bc_ranges[3]}]")


if __name__ == '__main__':
    main()
