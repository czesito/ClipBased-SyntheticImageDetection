#!/usr/bin/env python3
"""
Script to reorganize images based on classification results.
Clones images from data CSV and organizes them into real/fake folders
based on the fusion score in results CSV.

Features:
- Preserves source path information in output filenames (e.g., TISDC2025_fake_product.jpg)
- Ground truth determined by presence of 'real' in the filename path
- Configurable classification threshold
- Generates confusion matrix with performance metrics
- Detailed error analysis support

Usage:
    python reorganize_images.py --data_csv data/tisdc.csv --results_csv results.csv --output_dir data/output --threshold 0.0
"""

import argparse
import csv
import os
import re
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


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by replacing invalid characters with underscores.
    This preserves the path information from the CSV as part of the filename.
    
    Args:
        filename: Original filename (may contain path separators)
        
    Returns:
        Sanitized filename safe for filesystem
    """
    # Get the file extension first
    path_obj = Path(filename)
    extension = path_obj.suffix
    name_without_ext = str(path_obj.with_suffix(''))
    
    # Replace all invalid characters with underscore
    # Invalid chars: / \ : * ? " < > |
    # Also replace spaces for better compatibility
    sanitized = re.sub(r'[/\\:*?"<>|\s]', '_', name_without_ext)
    
    # Remove any leading/trailing underscores or dots
    sanitized = sanitized.strip('_.')
    
    # Combine back with extension
    return sanitized + extension


def is_real_image(filename: str) -> bool:
    """
    Determine if an image is real based on its filename path.
    If the path contains 'real' (case-insensitive), it's a real image.
    Otherwise, it's fake (regardless of the generation model).
    
    Args:
        filename: Original filename with path
        
    Returns:
        True if real image, False if fake
    """
    # Convert to lowercase for case-insensitive comparison
    filename_lower = filename.lower()
    return 'real' in filename_lower


def reorganize_images(data_csv: str, results_csv: str, output_dir: str, base_dir: str = 'data', threshold: float = 0.0):
    """
    Reorganize images based on fusion scores.
    
    Args:
        data_csv: Path to the data CSV file
        results_csv: Path to the results CSV file
        output_dir: Path to the output directory
        base_dir: Base directory where images are located (default: 'data')
        threshold: Threshold for classification. Fusion >= threshold is classified as fake (default: 0.0)
    """
    # Read the data
    print(f"Reading results from: {results_csv}")
    fusion_scores = read_results_csv(results_csv)
    
    print(f"Reading data from: {data_csv}")
    filenames = read_data_csv(data_csv)
    
    print(f"Using threshold: {threshold} (fusion >= {threshold} -> fake, fusion < {threshold} -> real)")

    output_path = Path(output_dir)

    # Delete existing output directory if it exists
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
        print(f"Deleted existing output directory: {output_path}")
    
    # Create output directories
    real_dir = output_path / 'real'
    fake_dir = output_path / 'fake'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"  - {real_dir}")
    print(f"  - {fake_dir}")
    
    # Statistics for confusion matrix
    # Ground truth: based on filename containing 'real'
    # Prediction: based on fusion score vs threshold
    true_positive = 0   # Correctly identified as fake
    true_negative = 0   # Correctly identified as real
    false_positive = 0  # Real image classified as fake
    false_negative = 0  # Fake image classified as real
    not_found = 0
    
    # List to store detailed results
    detailed_results = []
    
    # List to store detailed results
    detailed_results = []
    
        # Process each image
    for filename in filenames:
        # Get fusion score
        if filename not in fusion_scores:
            print(f"Warning: {filename} not found in results CSV")
            not_found += 1
            continue
        
        fusion_score = fusion_scores[filename]
        
        # Determine ground truth
        ground_truth_real = is_real_image(filename)
        ground_truth_label = 'real' if ground_truth_real else 'fake'
        
        # Determine prediction based on threshold
        predicted_real = fusion_score < threshold
        predicted_label = 'real' if predicted_real else 'fake'
        
        # Update confusion matrix statistics
        if ground_truth_real and predicted_real:
            true_negative += 1  # Real correctly identified as real
        elif ground_truth_real and not predicted_real:
            false_positive += 1  # Real incorrectly identified as fake
        elif not ground_truth_real and predicted_real:
            false_negative += 1  # Fake incorrectly identified as real
        else:  # not ground_truth_real and not predicted_real
            true_positive += 1  # Fake correctly identified as fake
        
        # Determine source and destination
        source_path = Path(base_dir) / filename
        
        # Create sanitized filename from CSV path (preserves source information)
        sanitized_name = sanitize_filename(filename)
        
        # Determine destination based on prediction (fusion score vs threshold)
        if predicted_real:
            dest_path = real_dir / sanitized_name
        else:
            dest_path = fake_dir / sanitized_name
        
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
            
            # Store detailed result
            detailed_results.append({
                'filename': filename,
                'fusion_score': fusion_score,
                'ground_truth': ground_truth_label,
                'prediction': predicted_label,
                'correct': ground_truth_label == predicted_label,
                'output_file': dest_path.name
            })
            
            correct_mark = '✓' if ground_truth_label == predicted_label else '✗'
            print(f"{correct_mark} GT:{ground_truth_label:4s} Pred:{predicted_label:4s} ({fusion_score:+.4f}): {filename} -> {dest_path.name}")
        else:
            print(f"Warning: Source file not found: {source_path}")
            not_found += 1
    
    # Generate confusion matrix CSV
    total_processed = true_positive + true_negative + false_positive + false_negative
    
    # Calculate metrics
    accuracy = (true_positive + true_negative) / total_processed if total_processed > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    confusion_matrix_path = output_path / 'confusion_matrix.csv'
    with open(confusion_matrix_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Confusion Matrix'])
        writer.writerow(['', 'Predicted Real', 'Predicted Fake'])
        writer.writerow(['Actual Real', true_negative, false_positive])
        writer.writerow(['Actual Fake', false_negative, true_positive])
        writer.writerow([])
        writer.writerow(['Metrics'])
        writer.writerow(['Accuracy', f'{accuracy:.4f}'])
        writer.writerow(['Precision', f'{precision:.4f}'])
        writer.writerow(['Recall', f'{recall:.4f}'])
        writer.writerow(['F1-Score', f'{f1_score:.4f}'])
        writer.writerow([])
        writer.writerow(['Detailed Statistics'])
        writer.writerow(['True Positive (Fake correctly identified)', true_positive])
        writer.writerow(['True Negative (Real correctly identified)', true_negative])
        writer.writerow(['False Positive (Real misclassified as Fake)', false_positive])
        writer.writerow(['False Negative (Fake misclassified as Real)', false_negative])
        writer.writerow(['Total Processed', total_processed])
        writer.writerow(['Not Found/Errors', not_found])
        writer.writerow([])
        writer.writerow(['Threshold', threshold])
    
    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images processed: {total_processed}")
    print(f"Threshold: {threshold}")
    print(f"\nPrediction Results:")
    print(f"  Predicted as Real: {true_negative + false_negative}")
    print(f"  Predicted as Fake: {true_positive + false_positive}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negative  (Real → Real): {true_negative}")
    print(f"  False Positive (Real → Fake): {false_positive}")
    print(f"  False Negative (Fake → Real): {false_negative}")
    print(f"  True Positive  (Fake → Fake): {true_positive}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1_score:.4f}")
    print(f"\nNot found/errors: {not_found}")
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Confusion matrix: {confusion_matrix_path.absolute()}")


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
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Classification threshold. Images with fusion >= threshold are classified as fake (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Reorganize images
    reorganize_images(
        data_csv=args.data_csv,
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        base_dir=args.base_dir,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
