# create_csv.py
import os
import pandas as pd
from pathlib import Path
import argparse

def create_dataset_csv(dataset_dir, output_csv):
    """
    Create a CSV file for all images in a directory.
    Expects structure: dataset_dir/real/ and dataset_dir/fake/
    """
    data = []
    
    # Process real images
    real_dir = os.path.join(dataset_dir, 'real')
    if os.path.exists(real_dir):
        for img in Path(real_dir).glob('**/*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                data.append({
                    'filename': str(img.relative_to(Path(dataset_dir).parent)),
                    'typ': 'real'
                })
    
    # Process fake images
    fake_dir = os.path.join(dataset_dir, 'fake')
    if os.path.exists(fake_dir):
        for img in Path(fake_dir).glob('**/*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                data.append({
                    'filename': str(img.relative_to(Path(dataset_dir).parent)),
                    'typ': 'fake'
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created CSV with {len(df)} images")
    print(f"Real: {len(df[df['typ']=='real'])}, Fake: {len(df[df['typ']=='fake'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a CSV file for image dataset')
    parser.add_argument('dataset_dir', type=str, help='Path to dataset directory (should contain real/ and fake/ subdirectories)')
    parser.add_argument('output_csv', type=str, help='Path to output CSV file')
    
    args = parser.parse_args()
    create_dataset_csv(args.dataset_dir, args.output_csv)