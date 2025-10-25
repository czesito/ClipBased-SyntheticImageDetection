"""
Training script for CLIP-based synthetic image detection.

This implements the methodology from:
"Raising the Bar of AI-generated Image Detection with CLIP"
Davide Cozzolino et al., 2023

Key points from the paper:
- Use CLIP ViT-L/14 as a frozen feature extractor
- Extract features from the next-to-last layer
- Train a linear SVM classifier on these features
- Use paired real/fake images with the same textual description
- Works with very small datasets (10-10k images)
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import argparse

from networks import create_architecture
from utils.processing import make_normalize


class ImageDataset:
    """Dataset for loading images and extracting CLIP features"""
    
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = os.path.dirname(os.path.abspath(csv_file))
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Label: 1 for fake, 0 for real
        label = 0 if self.data.iloc[idx]['typ'] == 'real' else 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def extract_features(model, dataset, device, batch_size=32):
    """
    Extract CLIP features from images.
    
    According to the paper, features are extracted from the next-to-last layer
    with the CLIP backbone frozen (no gradients).
    """
    features_list = []
    labels_list = []
    
    model = model.to(device)
    model.eval()
    
    # Process in batches for efficiency
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting features"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            
            batch_images = []
            batch_labels = []
            
            for idx in range(start_idx, end_idx):
                img, label = dataset[idx]
                batch_images.append(img)
                batch_labels.append(label)
            
            # Stack images into a batch
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Extract features using the forward_features method
            # This extracts from next-to-last layer with frozen CLIP
            features = model.forward_features(batch_tensor)
            
            features_list.append(features.cpu().numpy())
            labels_list.extend(batch_labels)
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.array(labels_list)
    
    return all_features, all_labels


def train_svm_classifier(train_features, train_labels, val_features=None, val_labels=None):
    """
    Train a linear SVM classifier on CLIP features.
    
    According to the paper:
    - Use linear SVM (not RBF or other kernels)
    - This is the standard approach described in Section 4
    """
    print("\nTraining linear SVM classifier...")
    
    # Create and train SVM with default parameters
    # The paper uses a simple linear SVM without mentioning specific hyperparameters
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(train_features, train_labels)
    
    # Evaluate on training set
    train_pred = svm.predict(train_features)
    train_acc = accuracy_score(train_labels, train_pred)
    print(f"Training Accuracy: {train_acc:.4f}")
    
    # Evaluate on validation set if provided
    if val_features is not None and val_labels is not None:
        val_pred = svm.predict(val_features)
        val_pred_proba = svm.predict_proba(val_features)[:, 1]
        
        val_acc = accuracy_score(val_labels, val_pred)
        val_auc = roc_auc_score(val_labels, val_pred_proba)
        
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
    
    return svm


def main():
    parser = argparse.ArgumentParser(description='Train CLIP-based synthetic image detector')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='Path to validation CSV file (optional)')
    parser.add_argument('--output_dir', type=str, default='weights/my_model',
                        help='Directory to save the trained model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for feature extraction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--arch', type=str, default='opencliplinearnext_clipL14commonpool',
                        help='Architecture name (should use next-to-last layer)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transform (matching the paper's preprocessing)
    # The paper uses CLIP's standard preprocessing:
    # - Resize to 224 with bicubic interpolation
    # - Center crop to 224x224
    # - CLIP normalization
    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop((224, 224)),
        make_normalize('clip')
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ImageDataset(args.train_csv, transform=transform)
    print(f"Training samples: {len(train_dataset)}")
    
    val_dataset = None
    if args.val_csv:
        val_dataset = ImageDataset(args.val_csv, transform=transform)
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    # IMPORTANT: Use 'opencliplinearnext_' to extract from next-to-last layer
    # as specified in the paper
    print(f"\nCreating model: {args.arch}")
    model = create_architecture(args.arch, num_classes=1)
    
    # Extract features from training set
    print("\nExtracting training features...")
    train_features, train_labels = extract_features(
        model, train_dataset, device, args.batch_size
    )
    
    # Extract features from validation set if provided
    val_features, val_labels = None, None
    if val_dataset:
        print("\nExtracting validation features...")
        val_features, val_labels = extract_features(
            model, val_dataset, device, args.batch_size
        )
    
    # Train SVM classifier
    svm_classifier = train_svm_classifier(
        train_features, train_labels,
        val_features, val_labels
    )
    
    # Save the trained SVM classifier
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save SVM classifier
    svm_path = os.path.join(args.output_dir, 'svm_classifier.pkl')
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_classifier, f)
    print(f"\nSaved SVM classifier to: {svm_path}")
    
    # Create config file
    config_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        f.write(f"arch: {args.arch}\n")
        f.write(f"model_name: {os.path.basename(args.output_dir)}\n")
        f.write(f"norm_type: clip\n")
        f.write(f"patch_size: Clip224\n")
        f.write(f"classifier_file: svm_classifier.pkl\n")
    print(f"Saved config to: {config_path}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"\nTo use this model for inference, you'll need to:")
    print(f"1. Extract CLIP features using the same architecture")
    print(f"2. Load the SVM classifier from: {svm_path}")
    print(f"3. Use svm.predict() or svm.decision_function() for classification")


if __name__ == '__main__':
    main()
