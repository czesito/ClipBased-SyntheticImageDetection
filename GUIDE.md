# Complete Guide: CLIP-Based Synthetic Image Detection

This comprehensive guide will walk you through everything you need to know to use this AI-generated image detection system based on CLIP features.

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [Downloading Required Files](#downloading-required-files)
5. [Quick Start: Testing Pre-trained Models](#quick-start-testing-pre-trained-models)
6. [Testing on Your Custom Dataset](#testing-on-your-custom-dataset)
7. [Fine-tuning on Your Custom Dataset](#fine-tuning-on-your-custom-dataset)
8. [Understanding the Output](#understanding-the-output)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## 🎯 Overview

This repository implements the paper "Raising the Bar of AI-generated Image Detection with CLIP". The system can detect AI-generated images from various generators (DALL-E, Midjourney, Stable Diffusion, etc.) with high accuracy.

**Key Features:**
- Detects images from 20+ different AI generators
- High generalization ability across different architectures
- Robust to image degradation and laundering
- Lightweight CLIP-based detector
- Pre-trained models ready to use

---

## 💻 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for datasets, ~1GB for model weights

### Software
- **Python**: 3.9 (recommended)
- **CUDA**: 11.8 or compatible version
- **Operating System**: Windows, Linux, or macOS

---

## 🔧 Installation & Setup

### Method 1: Conda Environment (Recommended)

1. **Install Anaconda/Miniconda** if you haven't already:
   - Download from: https://www.anaconda.com/download or https://docs.conda.io/en/latest/miniconda.html

2. **Create the environment from the provided YAML file:**
   ```bash
   conda env create -f environment.yaml
   ```

3. **Activate the environment:**
   ```bash
   conda activate clip-synthetic-detection
   ```

4. **Install additional Python packages:**
   ```bash
   pip install timm>=0.9.10
   pip install scikit-learn
   pip install pillow
   pip install pyyaml
   ```

### Method 2: pip Installation

If you prefer not to use Conda:

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

2. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install other dependencies:**
   ```bash
   pip install open_clip_torch huggingface_hub timm>=0.9.10
   pip install tqdm scikit-learn pillow pyyaml pandas numpy scipy
   ```

### Method 3: Docker

If you prefer using Docker:

1. **Build the Docker image:**
   ```bash
   docker build -t clipdet . -f Dockerfile
   ```

2. **Run the container** (see examples in Quick Start section)

---

## 📦 Downloading Required Files

### 1. Model Weights (Already Included)

The model weights should already be in the `weights/` directory:
- `weights/clipdet_latent10k_plus/` - Main model (recommended)
- `weights/clipdet_latent10k/` - Alternative model
- `weights/Corvi2023/` - Baseline comparison model

**If weights are missing**, download them using Git LFS:
```bash
git lfs install
git lfs pull
```

### 2. Synthbuster Dataset (For Commercial Tools Testing)

This dataset contains images from commercial AI generators (DALL-E 2, DALL-E 3, Midjourney v5, Firefly) and real images from the RAISE dataset.

**On Windows (PowerShell):**
```powershell
cd data
.\synthbuster_download.ps1
cd ..
```

**On Linux/Mac (Bash):**
```bash
cd data
bash synthbuster_download.sh
cd ..
```

**What gets downloaded:**
- `synthbuster.zip` (~8GB) - Synthetic images from commercial tools
- `real_RAISE_1k.zip` (~2GB) - 1000 real images

**After download, you'll have:**
```
data/
  synthbuster/
    dalle2/          # DALL-E 2 images
    dalle3/          # DALL-E 3 images
    midjourney5/     # Midjourney v5 images
    midjourney6/     # Midjourney v6 images
    firefly1/        # Adobe Firefly images
    real/            # Real RAISE images
```

### 3. Test Set (Optional - For Full Evaluation)

Download the comprehensive test set with 20 generators:

**Download link:** https://www.grip.unina.it/download/prog/DMimageDetection/test_set.zip
- **Size:** ~15GB
- **MD5:** `4a61d5185acc034e2a9884ccc3ff4e09`

**Extract to:**
```
data/test_set/
```

**Contents:** 4,000 real images + 20,000 synthetic images from 20 different generators

### 4. Training Set (Optional - For Fine-tuning)

Download the training dataset:

**Download link:** https://www.grip.unina.it/download/prog/DMimageDetection/train_set.zip
- **Size:** ~5GB
- **MD5:** `20febd0ca0a4ddb22848a0961b5c6c59`

**Extract to:**
```
data/train_set/
```

**Contents:** 10,000 real images + 10,000 fake images from LDM

---

## 🚀 Quick Start: Testing Pre-trained Models

### Example 1: Test on Commercial Tools Dataset

Once you've downloaded the Synthbuster dataset:

```bash
# Step 1: Run detection
python main.py --in_csv data/commercial_tools.csv --out_csv results.csv --device cuda:0

# Step 2: Compute metrics (AUC)
python compute_metrics.py --in_csv data/commercial_tools.csv --out_csv results.csv --metrics auc --save_tab auc_table.csv

# Step 3: View results
# The auc_table.csv will show AUC scores for each generator
```

**Expected output:** The model achieves high AUC scores (>0.95) on most commercial tools.

### Example 2: Using Docker

```bash
docker run --runtime=nvidia --gpus all \
  -v ${PWD}/data:/data_in \
  -v ${PWD}/:/data_out \
  clipdet \
  --in_csv /data_in/commercial_tools.csv \
  --out_csv /data_out/results.csv \
  --device cuda:0
```

### Example 3: CPU-only Mode

If you don't have a GPU:

```bash
python main.py --in_csv data/commercial_tools.csv --out_csv results.csv --device cpu
```

**Note:** CPU inference will be significantly slower (10-20x).

---

## 🎨 Testing on Your Custom Dataset

### Step 1: Prepare Your Images

Organize your images in a folder structure:

```
my_dataset/
  real/
    image1.jpg
    image2.png
    ...
  fake/
    image1.jpg
    image2.png
    ...
```

### Step 2: Create a CSV File

Create a CSV file (e.g., `my_images.csv`) with your image list:

**Format for detection only:**
```csv
filename
my_dataset/real/image1.jpg
my_dataset/real/image2.png
my_dataset/fake/image1.jpg
my_dataset/fake/image2.png
```

**Format for evaluation (with ground truth):**
```csv
filename,typ
my_dataset/real/image1.jpg,real
my_dataset/real/image2.png,real
my_dataset/fake/image1.jpg,fake_dalle
my_dataset/fake/image2.png,fake_midjourney
```

**Important notes:**
- The `filename` column is **required**
- Paths can be relative to the CSV file location or absolute
- The `typ` column is only needed if you want to compute metrics
- For real images, use `typ=real`
- For fake images, you can use any label (e.g., `fake`, `dalle3`, `midjourney`, etc.)

### Step 3: Generate a Python Script to Create the CSV

Here's a helper script to automatically generate the CSV:

```python
# create_csv.py
import os
import pandas as pd
from pathlib import Path

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

# Usage
create_dataset_csv('my_dataset', 'my_images.csv')
```

### Step 4: Run Detection

```bash
python main.py --in_csv my_images.csv --out_csv my_results.csv --device cuda:0
```

### Step 5: Analyze Results

```python
# analyze_results.py
import pandas as pd
import numpy as np

# Load results
results = pd.read_csv('my_results.csv')

# The 'fusion' column contains the final detection score (LLR - Log-Likelihood Ratio)
# LLR > 0 means the image is predicted as synthetic
# LLR < 0 means the image is predicted as real

print("Detection Summary:")
print(f"Total images: {len(results)}")
print(f"Predicted as synthetic (LLR > 0): {sum(results['fusion'] > 0)}")
print(f"Predicted as real (LLR < 0): {sum(results['fusion'] < 0)}")
print(f"\nScore statistics:")
print(results['fusion'].describe())

# If you have ground truth
if 'typ' in results.columns:
    results['predicted'] = results['fusion'] > 0
    results['actual'] = results['typ'] != 'real'
    
    correct = sum(results['predicted'] == results['actual'])
    accuracy = correct / len(results) * 100
    
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{len(results)}")
```

### Step 6: Compute Metrics (Optional)

If you included the `typ` column:

```bash
# Compute AUC
python compute_metrics.py --in_csv my_images.csv --out_csv my_results.csv --metrics auc --save_tab my_auc.csv

# Compute Balanced Accuracy
python compute_metrics.py --in_csv my_images.csv --out_csv my_results.csv --metrics acc --save_tab my_acc.csv
```

---

## 🎓 Fine-tuning on Your Custom Dataset

### Why Fine-tune?

Fine-tuning can improve performance when:
- Your images come from a specific domain or generator not well-represented in training
- You have a sufficient amount of labeled data (recommended: 1000+ images)
- You need to adapt to specific image characteristics or degradations

### Prerequisites

1. **Prepare your training data** in the same format as testing (see above)
2. **Split your data** into train/validation/test sets
3. **Ensure balanced classes** (similar number of real and fake images)

### Training Script

The repository doesn't include a training script by default, but here's a complete training pipeline:

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from networks import create_architecture
from utils.processing import make_normalize

class ImageDetectionDataset(Dataset):
    """Dataset for AI-generated image detection"""
    
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

def create_dataloaders(train_csv, val_csv, batch_size=32):
    """Create training and validation dataloaders"""
    
    # Define transforms (matching the pre-trained model)
    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop((224, 224)),
        make_normalize('clip')
    ])
    
    train_dataset = ImageDetectionDataset(train_csv, transform=transform)
    val_dataset = ImageDetectionDataset(val_csv, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, num_epochs=10, 
                learning_rate=1e-4):
    """Train the model"""
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            
            # Handle single output or binary output
            if len(outputs.shape) == 1:
                loss = criterion(outputs, labels)
                predictions = (torch.sigmoid(outputs) > 0.5).long()
            else:
                # If model outputs 2 values, use the difference
                outputs = outputs[:, 1] - outputs[:, 0]
                loss = criterion(outputs, labels)
                predictions = (torch.sigmoid(outputs) > 0.5).long()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (predictions == labels.long()).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': train_loss/train_total, 
                            'acc': train_correct/train_total})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images).squeeze()
                
                if len(outputs.shape) == 1:
                    loss = criterion(outputs, labels)
                    predictions = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    outputs = outputs[:, 1] - outputs[:, 0]
                    loss = criterion(outputs, labels)
                    predictions = (torch.sigmoid(outputs) > 0.5).long()
                
                val_loss += loss.item()
                val_correct += (predictions == labels.long()).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f'\nEpoch {epoch+1}:')
        print(f'  Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_correct/train_total:.4f}')
        print(f'  Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, 'best_model.pth')
            print(f'  → Saved best model (Val Acc: {val_acc:.4f})')
        
        scheduler.step()
        print()
    
    return model

if __name__ == '__main__':
    # Configuration
    TRAIN_CSV = 'data/train_set/train.csv'
    VAL_CSV = 'data/train_set/val.csv'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Create model
    print("Creating model...")
    model = create_architecture('opencliplinearnext_clipL14commonpool', 
                               num_classes=1)
    
    # Optionally load pre-trained weights
    # from networks import load_weights
    # model = load_weights(model, 'weights/clipdet_latent10k_plus/weights.pth')
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(TRAIN_CSV, VAL_CSV, BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Train
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, DEVICE, 
                       NUM_EPOCHS, LEARNING_RATE)
    
    print("\nTraining complete!")
```

### Fine-tuning Steps

1. **Prepare your CSV files:**
   - `train.csv` - Training set (80% of data)
   - `val.csv` - Validation set (10% of data)
   - `test.csv` - Test set (10% of data)

2. **Start with pre-trained weights (recommended):**
   ```python
   # In train.py, uncomment these lines:
   from networks import load_weights
   model = load_weights(model, 'weights/clipdet_latent10k_plus/weights.pth')
   ```

3. **Run training:**
   ```bash
   python train.py
   ```

4. **Test your fine-tuned model:**
   
   First, create a config file for your model:
   
   ```yaml
   # weights/my_model/config.yaml
   arch: opencliplinearnext_clipL14commonpool
   model_name: my_model
   norm_type: clip
   patch_size: Clip224
   weights_file: weights.pth
   ```
   
   Then copy your trained weights:
   ```bash
   mkdir -p weights/my_model
   cp best_model.pth weights/my_model/weights.pth
   ```
   
   Finally, test:
   ```bash
   python main.py --in_csv test.csv --out_csv test_results.csv \
                  --models my_model --device cuda:0
   ```

### Tips for Better Fine-tuning

1. **Data Augmentation:** Add augmentations to improve generalization
2. **Learning Rate:** Start with a smaller LR (1e-5) if fine-tuning from pre-trained weights
3. **Freeze Layers:** Consider freezing early layers and only training the classification head
4. **Class Balancing:** Use weighted loss if you have imbalanced classes
5. **Early Stopping:** Monitor validation loss to prevent overfitting

---

## 📊 Understanding the Output

### Output CSV Format

The output CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `filename` | Path to the image file |
| `clipdet_latent10k_plus` | Score from the main detector |
| `Corvi2023` | Score from the baseline detector |
| `fusion` | Combined score using soft-or fusion (default output) |

### Interpreting Scores

**LLR (Log-Likelihood Ratio) Score:**
- **LLR > 0**: Image predicted as **synthetic/AI-generated**
- **LLR < 0**: Image predicted as **real/authentic**
- **Larger absolute values** = higher confidence

**Examples:**
- `LLR = 5.2` → Strongly synthetic
- `LLR = 0.3` → Weakly synthetic
- `LLR = -0.8` → Weakly real
- `LLR = -4.1` → Strongly real

### Metrics

**AUC (Area Under ROC Curve):**
- Range: 0.0 to 1.0
- Perfect detector: 1.0
- Random detector: 0.5
- Higher is better

**Balanced Accuracy:**
- Average of sensitivity and specificity
- Range: 0.0 to 1.0
- Accounts for class imbalance

### Example Results Interpretation

```
                    dalle2  dalle3  midjourney5  firefly1   AVG
clipdet_latent10k_plus  0.986   0.992     0.978      0.985  0.985
Corvi2023              0.945   0.958     0.932      0.941  0.944
fusion                 0.991   0.995     0.983      0.989  0.990
```

This shows:
- The fusion model achieves ~99% AUC on average
- Performance is consistent across different generators
- DALL-E 3 is slightly easier to detect than others

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size (default is 1, should already be minimal)
- Use CPU mode: `--device cpu`
- Use a smaller image resolution (requires code modification)
- Close other GPU applications

#### 2. Module Not Found Errors

**Error:** `ModuleNotFoundError: No module named 'open_clip'`

**Solution:**
```bash
pip install open_clip_torch huggingface_hub
```

#### 3. Git LFS Files Not Downloaded

**Error:** Model weights are very small text files instead of large binary files

**Solution:**
```bash
git lfs install
git lfs pull
```

#### 4. Image Loading Errors

**Error:** `PIL.UnidentifiedImageError` or `Cannot identify image file`

**Solutions:**
- Ensure images are valid (not corrupted)
- Check supported formats: JPG, PNG, BMP, WEBP
- Try opening the image with PIL manually to verify

#### 5. CSV Parsing Errors

**Error:** `KeyError: 'filename'`

**Solution:**
- Ensure your CSV has a `filename` column header
- Check for proper CSV formatting
- Use comma as delimiter (not semicolon or tab)

#### 6. Path Issues on Windows

**Problem:** Paths with backslashes not working

**Solution:**
- Use forward slashes in CSV: `data/images/img.jpg`
- Or use raw strings in Python: `r"C:\path\to\image.jpg"`

#### 7. Slow Performance on CPU

**Problem:** Detection takes too long

**Solutions:**
- Use GPU if available
- Process images in batches
- Use the smaller model: `--models clipdet_latent10k`

---

## 🚀 Advanced Usage

### 1. Using Different Models

Test specific models:

```bash
# Use only the main model
python main.py --in_csv input.csv --out_csv output.csv --models clipdet_latent10k_plus

# Use only the baseline
python main.py --in_csv input.csv --out_csv output.csv --models Corvi2023

# Use all models
python main.py --in_csv input.csv --out_csv output.csv --models clipdet_latent10k_plus,clipdet_latent10k,Corvi2023
```

### 2. Changing Fusion Method

Different ways to combine model scores:

```bash
# Mean of logits (simple average)
python main.py --in_csv input.csv --out_csv output.csv --fusion mean_logit

# Maximum logit (most confident)
python main.py --in_csv input.csv --out_csv output.csv --fusion max_logit

# Soft-or probability (default, recommended)
python main.py --in_csv input.csv --out_csv output.csv --fusion soft_or_prob

# No fusion (individual model scores only)
python main.py --in_csv input.csv --out_csv output.csv --fusion None
```

**Available fusion methods:**
- `mean_logit` - Average of scores
- `max_logit` - Maximum score
- `median_logit` - Median score
- `lse_logit` - Log-sum-exp
- `mean_prob` - Average of probabilities
- `soft_or_prob` - Soft-OR of probabilities (default)

### 3. Batch Processing Script

Process multiple directories:

```python
# batch_process.py
import os
import subprocess
from pathlib import Path

def process_directory(img_dir, output_dir):
    """Process all images in a directory"""
    
    # Create CSV
    csv_file = os.path.join(output_dir, 'input.csv')
    with open(csv_file, 'w') as f:
        f.write('filename\n')
        for img in Path(img_dir).rglob('*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                f.write(f'{img}\n')
    
    # Run detection
    output_csv = os.path.join(output_dir, 'results.csv')
    cmd = [
        'python', 'main.py',
        '--in_csv', csv_file,
        '--out_csv', output_csv,
        '--device', 'cuda:0'
    ]
    subprocess.run(cmd)
    
    print(f"Results saved to: {output_csv}")

# Usage
process_directory('path/to/images', 'path/to/output')
```

### 4. Creating a Simple Web Interface

```python
# app.py - Simple Flask web interface
from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from PIL import Image
import io
import tempfile
import os

app = Flask(__name__)

# Load model (do this once at startup)
from networks import create_architecture, load_weights
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize

model = load_weights(
    create_architecture('opencliplinearnext_clipL14commonpool'),
    'weights/clipdet_latent10k_plus/weights.pth'
)
model = model.to('cuda:0').eval()

transform = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop((224, 224)),
    make_normalize('clip')
])

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>AI Image Detector</title></head>
    <body>
        <h1>AI-Generated Image Detector</h1>
        <form action="/detect" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Detect</button>
        </form>
    </body>
    </html>
    '''

@app.route('/detect', method=['POST'])
def detect():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    with torch.no_grad():
        img_tensor = transform(img).unsqueeze(0).to('cuda:0')
        output = model(img_tensor).cpu().numpy()[0]
        
        if len(output) == 1:
            score = output[0]
        else:
            score = output[1] - output[0]
    
    is_synthetic = score > 0
    confidence = abs(score)
    
    return jsonify({
        'is_synthetic': bool(is_synthetic),
        'score': float(score),
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### 5. Integration into Your Pipeline

```python
# detect.py - Standalone detection function
import torch
from PIL import Image
from networks import create_architecture, load_weights
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize

class AIImageDetector:
    def __init__(self, model_path='weights/clipdet_latent10k_plus/weights.pth', 
                 device='cuda:0'):
        self.device = device
        
        # Load model
        self.model = load_weights(
            create_architecture('opencliplinearnext_clipL14commonpool'),
            model_path
        )
        self.model = self.model.to(device).eval()
        
        # Setup transform
        self.transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((224, 224)),
            make_normalize('clip')
        ])
    
    def detect_image(self, image_path):
        """
        Detect if an image is AI-generated.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with keys: is_synthetic, score, confidence
        """
        img = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            output = self.model(img_tensor).cpu().numpy()[0]
            
            if len(output) == 1:
                score = output[0]
            else:
                score = output[1] - output[0]
        
        return {
            'is_synthetic': bool(score > 0),
            'score': float(score),
            'confidence': float(abs(score))
        }
    
    def detect_batch(self, image_paths):
        """Detect multiple images"""
        return [self.detect_image(path) for path in image_paths]

# Usage
if __name__ == '__main__':
    detector = AIImageDetector()
    result = detector.detect_image('test_image.jpg')
    
    if result['is_synthetic']:
        print(f"AI-generated (confidence: {result['confidence']:.2f})")
    else:
        print(f"Real image (confidence: {result['confidence']:.2f})")
```

---

## 📚 Additional Resources

### Citation

If you use this code or models in your research, please cite:

```bibtex
@inproceedings{cozzolino2023raising,
  author={Davide Cozzolino and Giovanni Poggi and Riccardo Corvi and Matthias Nießner and Luisa Verdoliva},
  title={{Raising the Bar of AI-generated Image Detection with CLIP}},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024},
}
```

### Links

- **Paper:** https://arxiv.org/abs/2312.00195v2
- **Project Page:** https://grip-unina.github.io/ClipBased-SyntheticImageDetection/
- **GRIP Lab:** https://www.grip.unina.it

### Dataset Citations

If using the Synthbuster dataset:

```bibtex
@article{bammey2023synthbuster,
  title={Synthbuster: Towards detection of diffusion model generated images},
  author={Bammey, Quentin},
  journal={IEEE Open Journal of Signal Processing},
  year={2023}
}

@inproceedings{dang2015raise,
  title={RAISE: A Raw Images Dataset for Digital Image Forensics},
  author={Dang-Nguyen, Duc-Tien and Pasquini, Cecilia and Conotter, Valentina and Boato, Giulia},
  booktitle={ACM MMSys},
  pages={219--224},
  year={2015}
}
```

---

## 📝 Summary

### Quick Reference Commands

```bash
# Setup
conda env create -f environment.yaml
conda activate clip-synthetic-detection

# Download data
cd data && .\synthbuster_download.ps1 && cd ..

# Test on commercial tools
python main.py --in_csv data/commercial_tools.csv --out_csv results.csv --device cuda:0
python compute_metrics.py --in_csv data/commercial_tools.csv --out_csv results.csv --metrics auc

# Test on custom dataset
python main.py --in_csv my_images.csv --out_csv my_results.csv --device cuda:0

# CPU mode
python main.py --in_csv input.csv --out_csv output.csv --device cpu
```

### Key Points to Remember

1. **LLR > 0** means AI-generated, **LLR < 0** means real
2. The `fusion` column in output is the final prediction
3. CSV must have a `filename` column
4. Add `typ` column for ground truth evaluation
5. Pre-trained models are in `weights/` directory
6. Use GPU for faster processing (`--device cuda:0`)

---

**Need Help?** Open an issue on the GitHub repository or contact the authors through the paper's contact information.

**License:** Apache License 2.0 - See LICENSE.md for details
