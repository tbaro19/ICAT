#!/bin/bash
# Complete setup script for ICAT project with local pyribs

echo "============================================================"
echo "ICAT Project Setup Script"
echo "============================================================"

# Activate conda environment
echo -e "\n1. Activating conda environment 'nlp'..."
source /root/miniconda3/bin/activate
conda activate nlp

# Install local pyribs
echo -e "\n2. Installing local pyribs from /root/ICAT/pyribs..."
cd /root/ICAT/pyribs
pip install -e . -q

# Install other dependencies
echo -e "\n3. Installing other dependencies..."
cd /root/ICAT
pip install -q torch torchvision transformers open_clip_torch timm
pip install -q bert-score sentence-transformers sacrebleu nltk
pip install -q underthesea pyvi
pip install -q matplotlib seaborn plotly shapely
pip install -q pandas pillow scikit-image opencv-python
pip install -q tqdm pyyaml wandb tensorboard

# Verify installations
echo -e "\n4. Verifying installations..."
python verify_pyribs.py

if [ $? -eq 0 ]; then
    echo -e "\n============================================================"
    echo "✅ Setup completed successfully!"
    echo "============================================================"
    echo ""
    echo "Quick start commands:"
    echo "  conda activate nlp"
    echo "  python test_datasets.py      # Test datasets"
    echo "  python verify_pyribs.py      # Verify pyribs"
    echo "  python demo.py               # Run quick demo"
    echo "  python main.py --help        # See all options"
    echo ""
else
    echo -e "\n❌ Setup failed. Please check errors above."
    exit 1
fi
