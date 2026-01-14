"""
Dataset loading utilities for KTVIC, UIT-ViIC, and Flickr30k
"""
import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable
import numpy as np


class ImageCaptionDataset(Dataset):
    """Generic image-caption dataset"""
    
    def __init__(self, 
                 images: List[str],
                 captions: List[str],
                 image_files: List[str],
                 transform: Optional[Callable] = None):
        """
        Initialize dataset
        
        Args:
            images: List of image paths or image tensors
            captions: List of captions
            image_files: List of image filenames
            transform: Transform to apply to images
        """
        self.images = images
        self.captions = captions
        self.image_files = image_files
        self.transform = transform
        
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        """Get item by index"""
        image_path = self.images[idx]
        caption = self.captions[idx]
        filename = self.image_files[idx]
        
        # Load image
        if isinstance(image_path, str):
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                else:
                    # Default: convert to tensor
                    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return black image as fallback
                image = torch.zeros(3, 224, 224)
        else:
            image = image_path
        
        return image, caption, filename


class DatasetLoader:
    """Loader for multiple Vietnamese and English image captioning datasets"""
    
    def __init__(self, data_root: str = '/root/ICAT/data'):
        """
        Initialize dataset loader
        
        Args:
            data_root: Root directory containing all datasets
        """
        self.data_root = data_root
        
    def load_ktvic(self, 
                   split: str = 'train',
                   transform: Optional[Callable] = None,
                   max_samples: Optional[int] = None) -> ImageCaptionDataset:
        """
        Load KTVIC (Vietnamese Life Domain) dataset
        
        Dataset structure:
        - train_data.json / test_data.json
        - train-images/ test-images/
        
        Args:
            split: 'train' or 'test'
            transform: Image transform function
            max_samples: Maximum number of samples to load
            
        Returns:
            ImageCaptionDataset instance
        """
        ktvic_dir = os.path.join(self.data_root, 'KTVIC', 'ktvic_dataset')
        
        # Load annotations
        json_file = os.path.join(ktvic_dir, f'{split}_data.json')
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse images and annotations
        images_info = {img['id']: img['filename'] for img in data['images']}
        
        # Determine image directory name
        if split == 'test':
            image_dir_name = 'public-test-images'
        else:
            image_dir_name = f'{split}-images'
        
        image_paths = []
        captions = []
        filenames = []
        
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in images_info:
                continue
            
            filename = images_info[image_id]
            image_path = os.path.join(ktvic_dir, image_dir_name, filename)
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
            
            image_paths.append(image_path)
            captions.append(ann['caption'])
            filenames.append(filename)
            
            if max_samples and len(image_paths) >= max_samples:
                break
        
        print(f"Loaded {len(image_paths)} KTVIC samples (split={split})")
        
        return ImageCaptionDataset(image_paths, captions, filenames, transform)
    
    def load_uit_viic(self,
                     split: str = 'train',
                     transform: Optional[Callable] = None,
                     max_samples: Optional[int] = None) -> ImageCaptionDataset:
        """
        Load UIT-ViIC (Vietnamese COCO) dataset
        
        Dataset structure:
        - uitvic_captions_train2017.json / uitvic_captions_test2017.json
        - coco_uitvic_train/ coco_uitvic_test/
        
        Args:
            split: 'train' or 'test'
            transform: Image transform function
            max_samples: Maximum number of samples to load
            
        Returns:
            ImageCaptionDataset instance
        """
        uitviic_dir = os.path.join(self.data_root, 'UIT-ViIC', 'uitvic_dataset')
        
        # Load annotations
        json_file = os.path.join(uitviic_dir, f'uitvic_captions_{split}2017.json')
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse images and annotations
        images_info = {img['id']: img['file_name'] for img in data['images']}
        
        # Image directory (nested structure)
        image_dir = os.path.join(uitviic_dir, f'coco_uitvic_{split}', f'coco_uitvic_{split}')
        
        image_paths = []
        captions = []
        filenames = []
        
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in images_info:
                continue
            
            filename = images_info[image_id]
            image_path = os.path.join(image_dir, filename)
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
            
            image_paths.append(image_path)
            captions.append(ann['caption'])
            filenames.append(filename)
            
            if max_samples and len(image_paths) >= max_samples:
                break
        
        print(f"Loaded {len(image_paths)} UIT-ViIC samples (split={split})")
        
        return ImageCaptionDataset(image_paths, captions, filenames, transform)
    
    def load_flickr30k(self,
                      transform: Optional[Callable] = None,
                      max_samples: Optional[int] = None,
                      comments_per_image: int = 1) -> ImageCaptionDataset:
        """
        Load Flickr30k dataset (English)
        
        Dataset structure:
        - flickr30k_images/results.csv
        - flickr30k_images/flickr30k_images/*.jpg
        
        Args:
            transform: Image transform function
            max_samples: Maximum number of samples to load
            comments_per_image: Number of captions per image to use (1-5)
            
        Returns:
            ImageCaptionDataset instance
        """
        flickr_dir = os.path.join(self.data_root, 'Flickr30k', 'flickr30k_images')
        
        # Load CSV with captions
        csv_file = os.path.join(flickr_dir, 'results.csv')
        df = pd.read_csv(csv_file, sep='|')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Image directory (nested structure)
        image_dir = os.path.join(flickr_dir, 'flickr30k_images')
        
        image_paths = []
        captions = []
        filenames = []
        
        # Group by image and take first N comments
        for image_name, group in df.groupby('image_name'):
            image_path = os.path.join(image_dir, image_name.strip())
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
            
            # Take first N comments
            for _, row in group.head(comments_per_image).iterrows():
                caption = row['comment'].strip()
                
                image_paths.append(image_path)
                captions.append(caption)
                filenames.append(image_name.strip())
                
                if max_samples and len(image_paths) >= max_samples:
                    break
            
            if max_samples and len(image_paths) >= max_samples:
                break
        
        print(f"Loaded {len(image_paths)} Flickr30k samples")
        
        return ImageCaptionDataset(image_paths, captions, filenames, transform)
    
    def load_coco(self,
                 split: str = 'val',
                 transform: Optional[Callable] = None,
                 max_samples: Optional[int] = None) -> ImageCaptionDataset:
        """
        Load MS-COCO dataset (if available)
        
        Args:
            split: 'train' or 'val'
            transform: Image transform function
            max_samples: Maximum number of samples to load
            
        Returns:
            ImageCaptionDataset instance
        """
        coco_dir = os.path.join(self.data_root, 'MS-COCO')
        
        if not os.path.exists(coco_dir):
            raise FileNotFoundError(
                f"MS-COCO dataset not found at {coco_dir}. "
                "Use KTVIC, UIT-ViIC, or Flickr30k instead."
            )
        
        # Load COCO annotations
        ann_file = os.path.join(coco_dir, 'annotations', f'captions_{split}2017.json')
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Parse images and annotations
        images_info = {img['id']: img['file_name'] for img in data['images']}
        image_dir = os.path.join(coco_dir, f'{split}2017')
        
        image_paths = []
        captions = []
        filenames = []
        
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in images_info:
                continue
            
            filename = images_info[image_id]
            image_path = os.path.join(image_dir, filename)
            
            if not os.path.exists(image_path):
                continue
            
            image_paths.append(image_path)
            captions.append(ann['caption'])
            filenames.append(filename)
            
            if max_samples and len(image_paths) >= max_samples:
                break
        
        print(f"Loaded {len(image_paths)} MS-COCO samples (split={split})")
        
        return ImageCaptionDataset(image_paths, captions, filenames, transform)


def create_dataloader(dataset: ImageCaptionDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4):
    """
    Create PyTorch DataLoader
    
    Args:
        dataset: ImageCaptionDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
