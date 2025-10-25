# CLIP-Based Synthetic Image Detection - User Guide

This guide provides instructions for using the AI-generated image detection system based on "Raising the Bar of AI-generated Image Detection with CLIP" (Cozzolino et al., 2024).

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Testing on Custom Dataset](#testing-on-custom-dataset)
4. [Training Your Own Detector](#training-your-own-detector)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yaml
conda activate clip-synthetic-detection
```

### Option 2: pip

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch huggingface_hub timm>=0.9.10
pip install tqdm scikit-learn pillow pyyaml pandas numpy scipy
```

---

## Quick Start

### 1. Download Test Data (Optional)

Download the Synthbuster dataset for testing:

```bash
cd data
# Linux/Mac:
bash synthbuster_download.sh
# Windows PowerShell:
.\synthbuster_download.ps1
cd ..
```

### 2. Run Detection

```bash
python main.py --in_csv data/commercial_tools.csv \
               --out_csv results.csv \
               --device cuda:0
```

### 3. Compute Metrics

```bash
python compute_metrics.py --in_csv data/commercial_tools.csv \
                          --out_csv results.csv \
                          --metrics auc \
                          --save_tab auc_table.csv
```

---

## Testing on Custom Dataset

### Step 1: Prepare CSV File

Create a CSV file listing your images:

```csv
filename,typ
path/to/real1.jpg,real
path/to/fake1.jpg,fake
path/to/real2.jpg,real
path/to/fake2.jpg,fake
```

- **filename**: Path to image (relative or absolute)
- **typ**: `real` for authentic images, `fake` for synthetic (or specify generator name)

### Step 2: Run Detection

```bash
python main.py --in_csv your_images.csv \
               --out_csv results.csv \
               --device cuda:0
```

### Step 3: Analyze Results

The output CSV contains a **fusion** column with detection scores (LLR):
- **LLR > 0**: Predicted as SYNTHETIC
- **LLR < 0**: Predicted as REAL
- Higher absolute value = higher confidence

```python
import pandas as pd
results = pd.read_csv('results.csv')
print(f"Synthetic: {(results['fusion'] > 0).sum()}")
print(f"Real: {(results['fusion'] < 0).sum()}")
```

---

## Training Your Own Detector

### Understanding the Methodology

**Important**: The paper uses a fundamentally different approach than traditional deep learning:

1. **Frozen CLIP backbone** - No backpropagation or weight updates
2. **Feature extraction** - Extract from next-to-last layer (1024-dim for ViT-L/14)
3. **Linear SVM classifier** - Train sklearn SVM on frozen features
4. **Minimal data** - Works with 10-10k images (optimal: 1k-10k)

This is NOT traditional fine-tuning! No epochs, no Adam optimizer, no BCELoss.

### Training Steps

#### 1. Prepare Training Data

Create train/val CSV files in the same format as above:

```csv
filename,typ
train/real1.jpg,real
train/fake1.jpg,fake
```

#### 2. Train SVM on CLIP Features

```bash
python train.py --train_csv data/train.csv \
                --val_csv data/val.csv \
                --output_dir weights/my_model \
                --device cuda:0
```

What happens:
- Extracts CLIP features (frozen backbone, no gradients)
- Trains linear SVM on features
- Saves to `weights/my_model/svm_classifier.pkl`

#### 3. Run Inference with Trained SVM

```bash
python inference_svm.py --svm_path weights/my_model/svm_classifier.pkl \
                        --in_csv test.csv \
                        --out_csv results.csv \
                        --device cuda:0
```

### Expected Performance

From the paper (Table based on reference set size N):

| N      | AUC (pristine) | AUC (post-processed) |
|--------|----------------|----------------------|
| 10     | ~90%           | ~70%                 |
| 100    | ~95%           | ~85%                 |
| 1,000  | ~97%           | ~90%                 |
| 10,000 | ~98%           | ~92%                 |

**Key insight**: Generalizes to unseen generators even when trained on a single generator!

### Implementation Details

```python
# Correct methodology (from paper Section 4)

# 1. Create frozen CLIP model
model = create_architecture('opencliplinearnext_clipL14commonpool', num_classes=1)
model.eval()

# 2. Extract features (no gradients)
with torch.no_grad():
    features = model.forward_features(images)  # Shape: (N, 1024)

# 3. Train linear SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear', probability=True)
svm.fit(features.numpy(), labels)

# 4. Save SVM
import pickle
with open('svm_classifier.pkl', 'wb') as f:
    pickle.dump(svm, f)
```

### Best Practices

✅ **DO:**
- Use `opencliplinearnext_*` architecture (next-to-last layer)
- Keep CLIP frozen (no backprop)
- Use 1k-10k images for training
- Train from single generator
- Use paired real/fake images for small datasets

❌ **DON'T:**
- Use Adam optimizer or gradient descent
- Use BCELoss or backpropagation
- Train for multiple epochs
- Use massive datasets (100k+)
- Apply heavy data augmentation

---

## Understanding Results

### Output Columns

| Column                    | Description                         |
|---------------------------|-------------------------------------|
| `filename`                | Path to the image file              |
| `clipdet_latent10k_plus`  | Score from main detector            |
| `Corvi2023`               | Score from baseline detector        |
| `fusion`                  | Combined score (default output)     |

### Score Interpretation

**LLR (Log-Likelihood Ratio):**
- LLR > 0: Synthetic/AI-generated
- LLR < 0: Real/authentic
- Larger |value| = higher confidence

**Examples:**
- `5.2` → Strongly synthetic
- `0.3` → Weakly synthetic
- `-0.8` → Weakly real
- `-4.1` → Strongly real

### Metrics

**AUC (Area Under ROC Curve):**
- Range: 0.0 to 1.0
- Perfect: 1.0, Random: 0.5

**Balanced Accuracy:**
- Average of sensitivity and specificity
- Accounts for class imbalance

---

## Advanced Usage

### Using Different Models

```bash
# Single model
python main.py --in_csv input.csv --out_csv output.csv \
               --models clipdet_latent10k_plus

# Multiple models
python main.py --in_csv input.csv --out_csv output.csv \
               --models clipdet_latent10k_plus,Corvi2023
```

### Fusion Methods

```bash
# Soft-or probability (default, recommended)
python main.py --fusion soft_or_prob ...

# Mean of logits
python main.py --fusion mean_logit ...

# Maximum logit
python main.py --fusion max_logit ...

# No fusion
python main.py --fusion None ...
```

Available methods: `mean_logit`, `max_logit`, `median_logit`, `lse_logit`, `mean_prob`, `soft_or_prob`

### CPU Mode

```bash
python main.py --in_csv input.csv --out_csv output.csv --device cpu
```

Note: CPU inference is 10-20x slower than GPU.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Use CPU mode
python main.py --device cpu --in_csv input.csv --out_csv output.csv
```

### Module Not Found

```bash
pip install open_clip_torch huggingface_hub timm scikit-learn
```

### Git LFS Files Not Downloaded

```bash
git lfs install
git lfs pull
```

### Image Loading Errors

- Check image formats: JPG, PNG, BMP, WEBP supported
- Verify images are not corrupted
- Ensure proper file paths

### CSV Parsing Errors

- Ensure CSV has `filename` column header
- Use comma as delimiter
- Check for proper formatting

### Slow Performance

- Use GPU if available
- Reduce batch size
- Use smaller model: `--models clipdet_latent10k`

---

## Citation

```bibtex
@inproceedings{cozzolino2023raising,
  author={Davide Cozzolino and Giovanni Poggi and Riccardo Corvi and 
          Matthias Nießner and Luisa Verdoliva},
  title={{Raising the Bar of AI-generated Image Detection with CLIP}},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition 
             Workshops (CVPRW)},
  year={2024},
}
```

---

## Additional Resources

- **Paper**: https://arxiv.org/abs/2312.00195v2
- **Project Page**: https://grip-unina.github.io/ClipBased-SyntheticImageDetection/
- **GRIP Lab**: https://www.grip.unina.it

---

## License

Apache License 2.0 - See LICENSE.md for details
