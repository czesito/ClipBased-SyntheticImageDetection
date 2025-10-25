#!/usr/bin/env python3
"""
Script to reorganize images based on classification results.
Clones images from data CSV and organizes them into real/fake folders
based on the fusion score in results CSV.

Usage:
    python reorganize_images.py --data_csv data/tisdc.csv --results_csv results.csv --output_dir output
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple


def read_results_csv(results_csv_path: str) -> Dict[str, float]:
    """
    Read results CSV and extract fusion scores for each image.
    
    Args:
        results_csv_path: Path to the results CSV file
        
    Returns:
        Dictionary mapping filename to fusion score
    """
    fusion_scores = {}
    
    with open(results_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            fusion = float(row['fusion'])
            fusion_scores[filename] = fusion
    
    return fusion_scores


def read_data_csv(data_csv_path: str) -> list:
    """
    Read data CSV to get list of image filenames.
    
    Args:
        data_csv_path: Path to the data CSV file
        
    Returns:
        List of filenames
    """
    filenames = []
    
    with open(data_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row['filename'])
    
    return filenames


def reorganize_images(data_csv: str, results_csv: str, output_dir: str, base_dir: str = 'data'):
    """
    Reorganize images based on fusion scores.
    
    Args:
        data_csv: Path to the data CSV file
        results_csv: Path to the results CSV file
        output_dir: Path to the output directory
        base_dir: Base directory where images are located (default: 'data')
    """
    # Read the data
    print(f"Reading results from: {results_csv}")
    fusion_scores = read_results_csv(results_csv)
    
    print(f"Reading data from: {data_csv}")
    filenames = read_data_csv(data_csv)
    
    # Create output directories
    output_path = Path(output_dir)
    real_dir = output_path / 'real'
    fake_dir = output_path / 'fake'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"  - {real_dir}")
    print(f"  - {fake_dir}")
    
    # Statistics
    real_count = 0
    fake_count = 0
    not_found = 0
    
    # Process each image
    for filename in filenames:
        # Get fusion score
        if filename not in fusion_scores:
            print(f"Warning: {filename} not found in results CSV")
            not_found += 1
            continue
        
        fusion_score = fusion_scores[filename]
        
        # Determine source and destination
        source_path = Path(base_dir) / filename
        
        # Get the original filename (without path)
        image_name = Path(filename).name
        
        # Determine destination based on fusion score
        if fusion_score < 0:
            # Real image
            dest_path = real_dir / image_name
            real_count += 1
        else:
            # Fake image
            dest_path = fake_dir / image_name
            fake_count += 1
        
        # Copy the image
        if source_path.exists():
            # Handle duplicate filenames by adding a suffix
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_path.parent / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(source_path, dest_path)
            print(f"{'Real' if fusion_score < 0 else 'Fake'} ({fusion_score:+.4f}): {filename} -> {dest_path.name}")
        else:
            print(f"Warning: Source file not found: {source_path}")
            not_found += 1
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(filenames)}")
    print(f"Real images (fusion < 0): {real_count}")
    print(f"Fake images (fusion >= 0): {fake_count}")
    print(f"Not found/errors: {not_found}")
    print(f"\nOutput directory: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize images based on classification results'
    )
    parser.add_argument(
        '--data_csv',
        type=str,
        default='data/tisdc.csv',
        help='Path to the data CSV file (default: data/tisdc.csv)'
    )
    parser.add_argument(
        '--results_csv',
        type=str,
        default='results.csv',
        help='Path to the results CSV file (default: results.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/output',
        help='Path to the output directory (default: output)'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='data',
        help='Base directory where images are located (default: data)'
    )
    
    args = parser.parse_args()
    
    # Reorganize images
    reorganize_images(
        data_csv=args.data_csv,
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        base_dir=args.base_dir
    )


if __name__ == '__main__':
    main()
