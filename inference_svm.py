"""
Inference script for SVM-based CLIP detector.

This script performs inference using an SVM classifier trained on CLIP features,
following the methodology from "Raising the Bar of AI-generated Image Detection with CLIP"
(Cozzolino et al., 2023).

Usage:
    python inference_svm.py --svm_path weights/my_model/svm_classifier.pkl \
                            --in_csv test.csv \
                            --out_csv results.csv \
                            --device cuda:0
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
import argparse

from networks import create_architecture
from utils.processing import make_normalize


def main():
    parser = argparse.ArgumentParser(description='Inference with SVM-based CLIP detector')
    parser.add_argument('--svm_path', type=str, required=True,
                        help='Path to trained SVM classifier (.pkl file)')
    parser.add_argument('--in_csv', type=str, required=True,
                        help='Path to input CSV file with image list')
    parser.add_argument('--out_csv', type=str, required=True,
                        help='Path to output CSV file for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for feature extraction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--arch', type=str, default='opencliplinearnext_clipL14commonpool',
                        help='Architecture name (must match training)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SVM classifier
    print(f"\nLoading SVM classifier from: {args.svm_path}")
    with open(args.svm_path, 'rb') as f:
        svm = pickle.load(f)
    
    # Create CLIP model for feature extraction
    print(f"Creating CLIP model: {args.arch}")
    model = create_architecture(args.arch, num_classes=1)
    model = model.to(device).eval()
    
    # Create transform
    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop((224, 224)),
        make_normalize('clip')
    ])
    
    # Load input CSV
    print(f"\nLoading images from: {args.in_csv}")
    df = pd.read_csv(args.in_csv)
    root_dir = os.path.dirname(os.path.abspath(args.in_csv))
    
    # Process images and extract features
    all_features = []
    all_filenames = []
    
    print("\nExtracting features...")
    num_batches = (len(df) + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(df))
            
            batch_images = []
            batch_filenames = []
            
            for idx in range(start_idx, end_idx):
                filename = df.iloc[idx]['filename']
                img_path = os.path.join(root_dir, filename)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = transform(image)
                    batch_images.append(img_tensor)
                    batch_filenames.append(filename)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # Extract features
            batch_tensor = torch.stack(batch_images).to(device)
            features = model.forward_features(batch_tensor).cpu().numpy()
            
            all_features.append(features)
            all_filenames.extend(batch_filenames)
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    
    # Run SVM inference
    print("\nRunning SVM inference...")
    predictions = svm.predict(all_features)  # 0=real, 1=fake
    probabilities = svm.predict_proba(all_features)  # [prob_real, prob_fake]
    decision_scores = svm.decision_function(all_features)  # SVM decision function (like LLR)
    
    # Create output dataframe
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'prediction': ['fake' if p == 1 else 'real' for p in predictions],
        'prob_real': probabilities[:, 0],
        'prob_fake': probabilities[:, 1],
        'decision_score': decision_scores
    })
    
    # Add original columns if they exist
    if 'typ' in df.columns:
        # Merge with original dataframe to get ground truth
        results_df = results_df.merge(df[['filename', 'typ']], on='filename', how='left')
        
        # Calculate accuracy
        results_df['correct'] = results_df.apply(
            lambda row: (row['prediction'] == 'real' and row['typ'] == 'real') or 
                       (row['prediction'] == 'fake' and row['typ'] != 'real'),
            axis=1
        )
        accuracy = results_df['correct'].mean()
        print(f"\nAccuracy: {accuracy:.4f}")
    
    # Save results
    print(f"\nSaving results to: {args.out_csv}")
    results_df.to_csv(args.out_csv, index=False)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total images processed: {len(results_df)}")
    print(f"Predicted as REAL: {(predictions == 0).sum()}")
    print(f"Predicted as FAKE: {(predictions == 1).sum()}")
    print(f"\nDecision score statistics:")
    print(f"  Mean: {decision_scores.mean():.4f}")
    print(f"  Std:  {decision_scores.std():.4f}")
    print(f"  Min:  {decision_scores.min():.4f}")
    print(f"  Max:  {decision_scores.max():.4f}")
    print("\nNote: Positive decision scores indicate FAKE, negative indicate REAL")
    

if __name__ == '__main__':
    main()
