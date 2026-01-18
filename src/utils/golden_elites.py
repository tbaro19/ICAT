"""
Golden Elite Extraction - Export top performing attacks with minimal perturbation
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from typing import List, Dict, Any


def export_golden_elites(
    archive,
    pert_gen,
    target_image: torch.Tensor,
    original_caption: str,
    groundtruth_caption: str,
    vlm_model,
    save_dir: str,
    num_elites: int = 5,
    fitness_weight: float = 0.7,
    bc1_weight: float = 0.3,
    generate_captions: bool = True
) -> List[str]:
    """
    Extract and export the top "Golden Elites" from the archive.
    
    Golden Elites are selected based on a weighted score that favors:
    - High fitness (attack effectiveness)
    - Low BC1 (L-inf norm - minimal visual perturbation)
    
    Selection Metric: Score = (fitness_weight * Normalized_Fitness) + (bc1_weight * (1 - Normalized_BC1))
    
    Args:
        archive: QDArchive containing elites
        pert_gen: PerturbationGenerator for converting solutions to perturbations
        target_image: Original high-res image tensor [C, H, W]
        original_caption: Original/generated caption for the clean image
        groundtruth_caption: Ground truth caption
        vlm_model: VLM model for caption generation (if needed)
        save_dir: Directory to save golden elite visualizations
        num_elites: Number of golden elites to extract (default: 5)
        fitness_weight: Weight for normalized fitness (default: 0.7)
        bc1_weight: Weight for inverted normalized BC1 (default: 0.3)
        generate_captions: Whether to generate captions for attacked images (default: True)
        
    Returns:
        List of saved file paths
    """
    print("\n" + "="*60)
    print("🏆 EXTRACTING GOLDEN ELITES")
    print("="*60)
    
    # Get all elites from archive
    all_elites = list(archive.archive)
    
    if len(all_elites) == 0:
        print("⚠️  No elites found in archive!")
        return []
    
    print(f"Total elites in archive: {len(all_elites)}")
    
    # Extract fitness and BC1 values
    fitness_values = np.array([elite['objective'] for elite in all_elites])
    bc1_values = np.array([elite['measures'][0] for elite in all_elites])
    bc2_values = np.array([elite['measures'][1] for elite in all_elites])
    
    # Normalize to [0, 1]
    fitness_min, fitness_max = fitness_values.min(), fitness_values.max()
    bc1_min, bc1_max = bc1_values.min(), bc1_values.max()
    
    # Avoid division by zero
    fitness_range = fitness_max - fitness_min if fitness_max > fitness_min else 1.0
    bc1_range = bc1_max - bc1_min if bc1_max > bc1_min else 1.0
    
    normalized_fitness = (fitness_values - fitness_min) / fitness_range
    normalized_bc1 = (bc1_values - bc1_min) / bc1_range
    
    # Compute weighted score (higher is better)
    # We invert BC1 because lower perturbation is better
    golden_scores = (fitness_weight * normalized_fitness) + (bc1_weight * (1 - normalized_bc1))
    
    # Select top N elites
    num_to_export = min(num_elites, len(all_elites))
    top_indices = np.argsort(golden_scores)[::-1][:num_to_export]
    
    print(f"\nSelection Metric: {fitness_weight:.1f}×Fitness + {bc1_weight:.1f}×(1-BC1)")
    print(f"Selecting top {num_to_export} golden elites...\n")
    
    # Print summary table
    print(f"{'Rank':<6} {'Fitness':<10} {'BC1 (L∞)':<12} {'BC2':<10} {'Golden Score':<12}")
    print("-" * 60)
    for rank, idx in enumerate(top_indices, 1):
        elite = all_elites[idx]
        print(f"{rank:<6} {elite['objective']:<10.4f} {elite['measures'][0]:<12.4f} "
              f"{elite['measures'][1]:<10.4f} {golden_scores[idx]:<12.4f}")
    
    # Create output directory
    golden_dir = os.path.join(save_dir, 'golden_elites')
    os.makedirs(golden_dir, exist_ok=True)
    
    saved_files = []
    
    # Process each golden elite
    for rank, idx in enumerate(top_indices, 1):
        elite = all_elites[idx]
        solution = elite['solution']
        fitness = elite['objective']
        bc1, bc2 = elite['measures']
        golden_score = golden_scores[idx]
        
        print(f"\n[{rank}/{num_to_export}] Processing Golden Elite #{rank}...")
        print(f"  Fitness: {fitness:.4f}, BC1: {bc1:.4f}, BC2: {bc2:.4f}, Score: {golden_score:.4f}")
        
        # Ensure target image is in correct format [C, H, W]
        if isinstance(target_image, np.ndarray):
            clean_image = torch.from_numpy(target_image).float()
        else:
            clean_image = target_image.clone()
        
        if clean_image.ndim == 2:
            clean_image = clean_image.unsqueeze(0).repeat(3, 1, 1)
        elif clean_image.ndim == 3 and clean_image.shape[2] == 3:
            clean_image = clean_image.permute(2, 0, 1)
        
        # Normalize to [0, 1]
        if clean_image.max() > 1.0:
            clean_image = clean_image / 255.0
        
        # Generate perturbation
        perturbation = pert_gen.solutions_to_perturbations(solution[np.newaxis, :])[0]
        perturbation = torch.from_numpy(perturbation).float()
        
        # Upsample perturbation to match image resolution
        H, W = clean_image.shape[1], clean_image.shape[2]
        perturbation_upsampled = torch.nn.functional.interpolate(
            perturbation.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Apply perturbation
        attacked_image = clean_image + perturbation_upsampled
        attacked_image = torch.clamp(attacked_image, 0, 1)
        
        # Generate caption for attacked image if requested
        attacked_caption = "N/A"
        if generate_captions:
            try:
                # Resize to model input size for caption generation
                attacked_resized = torch.nn.functional.interpolate(
                    attacked_image.unsqueeze(0),
                    size=(384, 384),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                from src.utils.comparison_viz import generate_caption_from_image, compute_image_text_score
                attacked_caption = generate_caption_from_image(vlm_model, attacked_resized)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Warning: Caption generation failed ({e})")
                # Use a meaningful placeholder instead of "Generation failed"
                attacked_caption = "Adversarial image (caption generation unavailable)"
        else:
            attacked_caption = "Caption generation disabled"
        
        # Compute all metrics like comparison_viz.py
        from bert_score import score as bert_score
        
        # BERTScore - handle model loading issues gracefully
        try:
            # Try with a more compatible model to avoid architecture mismatches
            P_att, R_att, F1_att = bert_score(
                [attacked_caption], [original_caption], 
                model_type='distilbert-base-uncased',  # Use smaller, more stable model
                lang='en', 
                verbose=False, 
                device='cpu'
            )
            bert_f1_att = F1_att.item()
            bert_f1_clean = 1.0
            bert_score_diff = bert_f1_clean - bert_f1_att
        except Exception as e:
            print(f"    Warning: BERTScore computation failed ({e}), using fallback values")
            bert_f1_clean = 1.0
            bert_f1_att = 0.0
            bert_score_diff = 0.0
        
        # Compute image-text scores
        try:
            clean_resized = torch.nn.functional.interpolate(
                clean_image.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False
            ).squeeze(0)
            attacked_resized_score = torch.nn.functional.interpolate(
                attacked_image.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            original_score = compute_image_text_score(vlm_model, clean_resized, groundtruth_caption)
            attacked_score = compute_image_text_score(vlm_model, attacked_resized_score, groundtruth_caption)
            score_diff = original_score - attacked_score
        except:
            original_score = 0.0
            attacked_score = 0.0
            score_diff = 0.0
        
        # WER (Word Error Rate)
        try:
            from jiwer import wer
            wer_score = wer(original_caption.lower(), attacked_caption.lower())
        except:
            wer_score = 0.0
        
        # CLIPScore
        try:
            clip_score_clean = compute_image_text_score(vlm_model, clean_resized, original_caption)
            clip_score_attacked = compute_image_text_score(vlm_model, clean_resized, attacked_caption)
        except:
            clip_score_clean = 0.0
            clip_score_attacked = 0.0
        
        # POS Divergence
        try:
            import spacy
            try:
                nlp = spacy.load('en_core_web_sm')
            except:
                nlp = None
            
            if nlp is not None:
                doc_clean = nlp(original_caption)
                doc_attacked = nlp(attacked_caption)
                
                nouns_clean = sum(1 for token in doc_clean if token.pos_ in ['NOUN', 'PROPN'])
                verbs_clean = sum(1 for token in doc_clean if token.pos_ == 'VERB')
                nouns_attacked = sum(1 for token in doc_attacked if token.pos_ in ['NOUN', 'PROPN'])
                verbs_attacked = sum(1 for token in doc_attacked if token.pos_ == 'VERB')
                
                total_clean = nouns_clean + verbs_clean
                total_attacked = nouns_attacked + verbs_attacked
                
                if total_clean > 0 or total_attacked > 0:
                    noun_diff = abs(nouns_clean - nouns_attacked)
                    verb_diff = abs(verbs_clean - verbs_attacked)
                    total_diff = noun_diff + verb_diff
                    max_total = max(total_clean, total_attacked)
                    pos_divergence = total_diff / max(max_total, 1)
                else:
                    pos_divergence = 0.0
            else:
                pos_divergence = 0.0
        except:
            pos_divergence = 0.0
        
        # Create visualization matching comparison_viz.py format
        fig = plt.figure(figsize=(32, 20))
        gs = GridSpec(5, 2, figure=fig, height_ratios=[0.72, 0.06, 0.06, 0.06, 0.10], 
                     hspace=0.12, wspace=0.08)
        
        # Title
        fig.suptitle(f'🏆 Golden Elite #{rank} (Score: {golden_score:.4f})', fontsize=18, fontweight='bold', y=0.98)
        
        # Original image (top left)
        ax_img_orig = fig.add_subplot(gs[0, 0])
        ax_img_orig.imshow(clean_image.permute(1, 2, 0).cpu().numpy())
        ax_img_orig.set_title('Clean Image', fontsize=16, fontweight='bold', pad=10)
        ax_img_orig.axis('off')
        
        # Attacked image (top right)
        ax_img_attacked = fig.add_subplot(gs[0, 1])
        ax_img_attacked.imshow(attacked_image.permute(1, 2, 0).cpu().numpy())
        ax_img_attacked.set_title('Attacked Image', fontsize=16, fontweight='bold', pad=10)
        ax_img_attacked.axis('off')
        
        # Add BC values as text overlay on attacked image
        bc_text = f"BC1 (L∞): {bc1:.4f}\nBC2: {bc2:.4f}"
        ax_img_attacked.text(
            0.02, 0.98, bc_text,
            transform=ax_img_attacked.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, pad=0.5),
            fontweight='bold'
        )
        
        # Original caption (below original image)
        ax_cap_orig = fig.add_subplot(gs[1, 0])
        ax_cap_orig.axis('off')
        import textwrap
        wrapped_orig = textwrap.fill(original_caption, width=50)
        ax_cap_orig.text(
            0.5, 0.5, 
            wrapped_orig,
            ha='center', va='center',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.8),
            transform=ax_cap_orig.transAxes
        )
        
        # Attacked caption (below attacked image)
        ax_cap_gen = fig.add_subplot(gs[1, 1])
        ax_cap_gen.axis('off')
        wrapped_gen = textwrap.fill(attacked_caption, width=50)
        ax_cap_gen.text(
            0.5, 0.5,
            wrapped_gen,
            ha='center', va='center',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, pad=0.8),
            transform=ax_cap_gen.transAxes
        )
        
        # BERTScore (spanning both columns)
        ax_bert = fig.add_subplot(gs[2, :])
        ax_bert.axis('off')
        ax_bert.text(
            0.5, 0.5,
            f"BERTScore: Clean={bert_f1_clean:.4f}, Attacked={bert_f1_att:.4f}, Δ={bert_score_diff:.4f} | Score Diff (Orig-Attack): {score_diff:.4f}",
            ha='center', va='center',
            fontsize=13,
            fontweight='bold',
            color='blue',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='blue', linewidth=2, pad=0.8),
            transform=ax_bert.transAxes
        )
        
        # WER and CLIPScore (spanning both columns)
        ax_wer = fig.add_subplot(gs[3, :])
        ax_wer.axis('off')
        ax_wer.text(
            0.5, 0.5,
            f"WER: {wer_score:.4f} | CLIPScore: Clean={clip_score_clean:.4f}, Attacked={clip_score_attacked:.4f}, Δ={clip_score_clean - clip_score_attacked:.4f}",
            ha='center', va='center',
            fontsize=13,
            fontweight='bold',
            color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2, pad=0.8),
            transform=ax_wer.transAxes
        )
        
        # POS Divergence and Golden Score (spanning both columns)
        ax_pos = fig.add_subplot(gs[4, :])
        ax_pos.axis('off')
        ax_pos.text(
            0.5, 0.5,
            f"POS Divergence: {pos_divergence:.4f} | 🏆 Golden Score: {fitness_weight:.1f}×Norm(F) + {bc1_weight:.1f}×(1-BC1) = {golden_score:.4f} | Fitness: {fitness:.4f}",
            ha='center', va='center',
            fontsize=13,
            fontweight='bold',
            color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender', edgecolor='purple', linewidth=2, pad=0.8),
            transform=ax_pos.transAxes
        )
        
        # Save figure
        filename = f"golden_elite_{rank:02d}_fitness{fitness:.3f}_bc1{bc1:.3f}.png"
        filepath = os.path.join(golden_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        saved_files.append(filepath)
        print(f"  ✓ Saved: {filename}")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create summary figure
    print(f"\n📊 Creating summary visualization...")
    summary_fig, axes = plt.subplots(1, num_to_export, figsize=(4*num_to_export, 5))
    if num_to_export == 1:
        axes = [axes]
    
    for rank, (idx, ax) in enumerate(zip(top_indices, axes), 1):
        elite = all_elites[idx]
        solution = elite['solution']
        fitness = elite['objective']
        bc1, bc2 = elite['measures']
        
        # Generate attacked image
        perturbation = pert_gen.solutions_to_perturbations(solution[np.newaxis, :])[0]
        perturbation = torch.from_numpy(perturbation).float()
        
        clean_image = target_image.clone() if isinstance(target_image, torch.Tensor) else torch.from_numpy(target_image).float()
        if clean_image.max() > 1.0:
            clean_image = clean_image / 255.0
        
        H, W = clean_image.shape[1], clean_image.shape[2]
        perturbation_upsampled = torch.nn.functional.interpolate(
            perturbation.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        attacked_image = torch.clamp(clean_image + perturbation_upsampled, 0, 1)
        
        # Display side-by-side
        combined = torch.cat([clean_image, attacked_image], dim=2)
        ax.imshow(combined.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"#{rank}\nF={fitness:.3f}, L∞={bc1:.3f}\nScore={golden_scores[idx]:.3f}", 
                    fontsize=9)
        ax.axis('off')
    
    summary_fig.suptitle(f"Top {num_to_export} Golden Elites (Clean | Attacked)", fontsize=12, weight='bold')
    summary_path = os.path.join(golden_dir, 'golden_elites_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(summary_fig)
    saved_files.append(summary_path)
    print(f"  ✓ Saved: golden_elites_summary.png")
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully exported {num_to_export} golden elites")
    print(f"📁 Directory: {golden_dir}")
    print(f"{'='*60}\n")
    
    return saved_files
