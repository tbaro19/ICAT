#!/usr/bin/env python3
"""Extract Flickr30k dataset"""
import zipfile
import os
from tqdm import tqdm

zip_path = '/root/ICAT/data/Flickr30k/flickr-image-dataset.zip'
extract_dir = '/root/ICAT/data/Flickr30k/'

print(f"Extracting {zip_path}...")
print(f"To: {extract_dir}")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Get list of files
    file_list = zip_ref.namelist()
    print(f"Total files: {len(file_list)}")
    
    # Extract with progress bar
    for file in tqdm(file_list, desc="Extracting"):
        zip_ref.extract(file, extract_dir)

print("\nExtraction complete!")
print("\nContents:")
os.system(f"ls -lh {extract_dir}")
