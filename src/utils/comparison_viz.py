"""
Create before/after attack comparison visualizations
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
from typing import Tuple


def create_attack_comparison(
    original_image: np.ndarray,
    attacked_image: np.ndarray,
    original_caption: str,
    attacked_caption: str,
    original_score: float,
    attacked_score: float,
    save_path: str,
    caption_similarity_diff: float = None,
    bert_score_diff: float = None,
    bert_f1_orig: float = None,
    bert_f1_gen: float = None,
    bc1_value: float = None,
    bc2_value: float = None,
    bc1_name: str = "BC1",
    bc2_name: str = "BC2",
    wer_score: float = None,
    clip_score_clean: float = None,
    clip_score_attacked: float = None,
    pos_divergence: float = None
):
    """
    Create a comparison visualization showing before/after attack
    
    Args:
        original_image: Original image [C, H, W] in [0, 1]
        attacked_image: Attacked image [C, H, W] in [0, 1]
        original_caption: Caption generated from original image
        attacked_caption: Caption generated from attacked image
        original_score: Similarity score for original
        attacked_score: Similarity score for attacked
        save_path: Path to save visualization
        bc1_value: First behavior characteristic value
        bc2_value: Second behavior characteristic value
        bc1_name: Name of first BC
        bc2_name: Name of second BC
    """
    # Convert torch tensors to numpy if needed
    if torch.is_tensor(original_image):
        original_image = original_image.cpu().numpy()
    if torch.is_tensor(attacked_image):
        attacked_image = attacked_image.cpu().numpy()
    
    # Debug: Print image shape
    print(f"  Visualization image shape: {original_image.shape}")
    
    # Convert from [C, H, W] to [H, W, C]
    if original_image.shape[0] == 3:
        original_image = original_image.transpose(1, 2, 0)
    if attacked_image.shape[0] == 3:
        attacked_image = attacked_image.transpose(1, 2, 0)
    
    # Clip to valid range
    original_image = np.clip(original_image, 0, 1)
    attacked_image = np.clip(attacked_image, 0, 1)
    
    print(f"  Final visualization image shape (HWC): {original_image.shape}")
    
    # Create figure with GridSpec layout - 5 rows: images + 4 metric rows
    fig = plt.figure(figsize=(32, 20))
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[0.72, 0.06, 0.06, 0.06, 0.10], 
                          hspace=0.12, wspace=0.08)
    
    # Title
    fig.suptitle('Attack Example', fontsize=18, fontweight='bold', y=0.98)
    
    # Original image (top left)
    ax_img_orig = fig.add_subplot(gs[0, 0])
    ax_img_orig.imshow(original_image, interpolation='none')
    ax_img_orig.set_title('Original Image', fontsize=16, fontweight='bold', pad=10)
    ax_img_orig.axis('off')
    
    # Attacked image (top right)
    ax_img_attacked = fig.add_subplot(gs[0, 1])
    ax_img_attacked.imshow(attacked_image, interpolation='none')
    ax_img_attacked.set_title('Attacked Image', fontsize=16, fontweight='bold', pad=10)
    ax_img_attacked.axis('off')
    
    # Add BC values as text overlay on attacked image
    if bc1_value is not None and bc2_value is not None:
        bc_text = f"{bc1_name}: {bc1_value:.4f}\n{bc2_name}: {bc2_value:.4f}"
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
    wrapped_orig = textwrap.fill(original_caption, width=50)
    ax_cap_orig.text(
        0.5, 0.5, 
        wrapped_orig,
        ha='center', va='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.8),
        transform=ax_cap_orig.transAxes
    )
    
    # Generated caption (below attacked image)
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
    if bert_f1_orig is not None and bert_f1_gen is not None and bert_score_diff is not None:
        ax_bert = fig.add_subplot(gs[2, :])
        ax_bert.axis('off')
        score_diff = original_score - attacked_score
        ax_bert.text(
            0.5, 0.5,
            f"BERTScore: Clean={bert_f1_orig:.4f}, Attacked={bert_f1_gen:.4f}, Δ={bert_score_diff:.4f} | Score Diff (Orig-Attack): {score_diff:.4f}",
            ha='center', va='center',
            fontsize=13,
            fontweight='bold',
            color='blue',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='blue', linewidth=2, pad=0.8),
            transform=ax_bert.transAxes
        )
    
    # WER and CLIPScore (spanning both columns)
    if wer_score is not None and clip_score_clean is not None and clip_score_attacked is not None:
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
    
    # POS Divergence (spanning both columns)
    if pos_divergence is not None:
        ax_pos = fig.add_subplot(gs[4, :])
        ax_pos.axis('off')
        ax_pos.text(
            0.5, 0.5,
            f"POS Divergence: {pos_divergence:.4f}",
            ha='center', va='center',
            fontsize=13,
            fontweight='bold',
            color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender', edgecolor='purple', linewidth=2, pad=0.8),
            transform=ax_pos.transAxes
        )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attack comparison saved to {save_path}")


def generate_caption_from_image(vlm_model, image: np.ndarray, language: str = 'en', max_length: int = 50) -> str:
    """
    Generate a caption from an image using the VLM model (BLIP-2 or LLaVA)
    
    Args:
        vlm_model: VLM model instance (BLIP2Wrapper or LLaVAWrapper)
        image: Image array [C, H, W] in [0, 1]
        language: Language for captions ('en' for English, 'vi' for Vietnamese)
        max_length: Maximum caption length
        
    Returns:
        Generated caption string
    """
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image
    
    # Add batch dimension if needed
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Generate caption using the model's generative capabilities
    with torch.no_grad():
        captions = vlm_model.generate_caption(image_tensor, max_length=max_length)
    
    # Return first caption (single image)
    caption = captions[0] if isinstance(captions, list) else captions
    
    # Note: For Vietnamese, we can translate the English caption if needed
    if language == 'vi':
        from src.utils.translation import translate_en_to_vi
        caption = translate_en_to_vi(caption)
    
    return caption


def compute_image_text_score(vlm_model, image: np.ndarray, text: str) -> float:
    """
    Compute similarity score between image and text
    
    Args:
        vlm_model: VLM model instance
        image: Image array [C, H, W] in [0, 1]
        text: Text caption
        
    Returns:
        Similarity score
    """
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image
    
    # Add batch dimension if needed
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Compute similarity
    with torch.no_grad():
        similarity = vlm_model.compute_similarity(image_tensor, [text])
    
    return similarity.item()
