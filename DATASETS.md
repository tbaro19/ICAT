# Dataset Information

## Available Datasets

All 3 datasets have been downloaded and verified in `/root/ICAT/data/`:

### 1. KTVIC (Vietnamese Life Domain)
- **Location**: `/root/ICAT/data/KTVIC/ktvic_dataset/`
- **Samples**: 2,790 test images
- **Language**: Vietnamese
- **Format**: JSON annotations with image directory
- **Source**: https://www.kaggle.com/datasets/leo040802/ktvic-dataset
- **Description**: Vietnamese life domain images with natural Vietnamese captions

**Usage:**
```python
from src.utils.data_loader import DatasetLoader

loader = DatasetLoader()
dataset = loader.load_ktvic(split='test', max_samples=100)
```

### 2. UIT-ViIC (Vietnamese COCO)
- **Location**: `/root/ICAT/data/UIT-ViIC/uitvic_dataset/`
- **Samples**: 1,155 test images  
- **Language**: Vietnamese
- **Format**: COCO-style JSON annotations
- **Source**: https://www.kaggle.com/datasets/leo040802/uitvic-dataset
- **Description**: Vietnamese translation of COCO dataset

**Usage:**
```python
dataset = loader.load_uit_viic(split='test', max_samples=100)
```

### 3. Flickr30k (English)
- **Location**: `/root/ICAT/data/Flickr30k/flickr30k_images/`
- **Samples**: 31,783 images (with 5 captions each)
- **Language**: English
- **Format**: CSV file with image-caption pairs
- **Source**: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
- **Description**: Standard English image captioning benchmark

**Usage:**
```python
dataset = loader.load_flickr30k(max_samples=100, comments_per_image=1)
```

## Dataset Statistics

| Dataset | Language | Images | Captions | Domain |
|---------|----------|--------|----------|--------|
| KTVIC | Vietnamese | 2,790 | ~13,950 | Life scenes |
| UIT-ViIC | Vietnamese | 1,155 | ~5,775 | COCO objects |
| Flickr30k | English | 31,783 | ~158,915 | General scenes |

## Verification

Run the verification script to check all datasets:

```bash
python test_datasets.py
```

Or create visualizations:

```bash
python verify_datasets.py
```

This will generate `/root/ICAT/results/dataset_samples.png` showing samples from each dataset.

## Dataset Structure

### KTVIC
```
KTVIC/
└── ktvic_dataset/
    ├── train_data.json          # Training annotations
    ├── test_data.json           # Test annotations
    ├── train-images/            # Training images
    └── public-test-images/      # Test images
```

**JSON Format:**
```json
{
  "images": [{"id": 1, "filename": "00000000001.jpg"}],
  "annotations": [
    {
      "id": 0,
      "image_id": 1,
      "caption": "ba chiếc thuyền đang di chuyển ở trên con sông"
    }
  ]
}
```

### UIT-ViIC
```
UIT-ViIC/
└── uitvic_dataset/
    ├── uitvic_captions_train2017.json
    ├── uitvic_captions_test2017.json
    ├── coco_uitvic_train/
    │   └── coco_uitvic_train/   # Nested directory
    └── coco_uitvic_test/
        └── coco_uitvic_test/    # Nested directory
```

**JSON Format:**
```json
{
  "images": [
    {
      "id": 535668,
      "file_name": "000000535668.jpg",
      "height": 426,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 4990,
      "image_id": 157656,
      "caption": "Người đàn ông đang đánh tennis ngoài sân."
    }
  ]
}
```

### Flickr30k
```
Flickr30k/
└── flickr30k_images/
    ├── results.csv              # All captions
    └── flickr30k_images/        # Nested directory with images
```

**CSV Format:**
```
image_name| comment_number| comment
1000092795.jpg| 0| Two young guys with shaggy hair look...
1000092795.jpg| 1| Two young , White males are outside...
```

## Using Datasets in Experiments

### Single Dataset Experiment

```bash
# Attack KTVIC (Vietnamese)
python main.py --dataset ktvic --max_samples 100 --iterations 500

# Attack UIT-ViIC (Vietnamese)
python main.py --dataset uit-viic --max_samples 100 --iterations 500

# Attack Flickr30k (English)
python main.py --dataset flickr30k --max_samples 100 --iterations 500
```

### Cross-Dataset Validation

Test the same attack across different datasets:

```bash
# Run on all datasets
for dataset in ktvic uit-viic flickr30k; do
    python main.py \
        --dataset $dataset \
        --model siglip \
        --algorithm cma_me \
        --iterations 500 \
        --max_samples 50 \
        --exp_name cross_dataset_${dataset}
done
```

### Language Comparison

Compare Vietnamese vs English vulnerability:

```bash
# Vietnamese datasets
python main.py --dataset ktvic --exp_name vietnamese_ktvic
python main.py --dataset uit-viic --exp_name vietnamese_uitviic

# English dataset
python main.py --dataset flickr30k --exp_name english_flickr30k
```

## Dataset Loading API

The `DatasetLoader` class provides a unified interface:

```python
from src.utils.data_loader import DatasetLoader

# Initialize
loader = DatasetLoader(data_root='/root/ICAT/data')

# Load any dataset
dataset = loader.load_ktvic(split='test', max_samples=100)
dataset = loader.load_uit_viic(split='test', max_samples=100)
dataset = loader.load_flickr30k(max_samples=100, comments_per_image=1)

# Access samples
image, caption, filename = dataset[0]

# Image: torch.Tensor [C, H, W] in range [0, 1]
# Caption: str (Vietnamese or English)
# Filename: str (original filename)

# Get dataset size
print(f"Dataset has {len(dataset)} samples")
```

## Notes

- All images are automatically converted to RGB format
- Images are returned as PyTorch tensors in range [0, 1]
- Vietnamese captions use UTF-8 encoding
- Flickr30k has 5 captions per image (use `comments_per_image` to control)
- KTVIC and UIT-ViIC have ~5 captions per image on average
