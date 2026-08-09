# 🩺 Diabetic Retinopathy Classification using MobileNetV2

Deep learning project for **5-class diabetic retinopathy severity classification** from retinal fundus images using **PyTorch** and a pretrained **MobileNetV2** backbone.

## 📌 Overview

The project uses the **APTOS 2019** dataset to classify retinal fundus images into five diabetic retinopathy severity grades:

| Class | Severity |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

The model uses ImageNet-pretrained MobileNetV2 and progressively fine-tunes the backbone.

## ✨ Features

- 🧠 Transfer learning with pretrained MobileNetV2
- 🔄 Progressive layer unfreezing
- 🖼️ Image augmentation and ImageNet normalization
- ⚡ CUDA / MPS / CPU support
- 📊 Training and validation metrics
- 🧮 Confusion matrix and classification report
- 🔍 Grad-CAM visualization
- 📉 t-SNE feature visualization
- 💾 Model checkpoint saving

## 📂 Dataset

APTOS 2019 Blindness Detection dataset:

https://www.kaggle.com/datasets/mariaherrerot/aptos2019

Expected structure:

```text
data/
└── aptos2019/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── train_images/
    ├── val_images/
    └── test_images/
```

The notebook contains **2,930 training samples** and **366 validation samples**. The test split contains **366 labels**.

## 🧠 Model Architecture

A pretrained MobileNetV2 backbone is used with a custom classifier:

```text
MobileNetV2 Backbone
        │
        ▼
    Dropout (0.2)
        │
        ▼
  Linear (1280 → 512)
        │
        ▼
       ReLU
        │
        ▼
    Dropout (0.2)
        │
        ▼
  Linear (512 → 5)
```

**Total parameters:** 2,882,309

## 🔄 Progressive Fine-Tuning

Training is divided into four stages:

| Stage | Trainable Parameters | Learning Rate | Epochs |
|---|---:|---:|---:|
| Classifier only | 658,437 | 1e-3 | 15 |
| Last 4 layers | 2,184,517 | 5e-4 | 15 |
| Last 8 layers | 2,642,949 | 1e-4 | 15 |
| Full fine-tuning | 2,882,309 | 5e-5 | 20 |

`ReduceLROnPlateau` scheduling and early stopping are used during training.

## 🖼️ Preprocessing

Images are resized to **224 × 224**.

Training augmentation:
- Random horizontal flip
- Random rotation up to 10°
- Color jitter
- ImageNet normalization

Validation/test images use resizing and normalization without random augmentation.

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 128 |
| Loss | Cross Entropy |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Scheduler | ReduceLROnPlateau |
| Backbone | MobileNetV2 |
| Number of Classes | 5 |

## 📊 Results

Final validation performance recorded in the notebook:

| Metric | Score |
|---|---:|
| Validation Accuracy | **84.15%** |
| Validation Loss | **0.9348** |
| Macro F1 | **0.6869** |
| Weighted F1 | **0.8319** |

### Classification Report

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No DR | 0.9771 | 0.9942 | 0.9856 |
| Mild | 0.7647 | 0.6500 | 0.7027 |
| Moderate | 0.7317 | 0.8654 | 0.7930 |
| Severe | 0.6667 | 0.2727 | 0.3871 |
| Proliferative DR | 0.6000 | 0.5357 | 0.5660 |

> These are validation-set results from the notebook and should not be interpreted as clinical performance.

## 📈 Evaluation & Visualization

The notebook includes:

- Training/validation loss curves
- Training/validation accuracy curves
- Stage-wise validation accuracy
- Learning-rate schedule
- Confusion matrix
- Per-class accuracy
- Grad-CAM visualization
- t-SNE feature visualization

## 💾 Model Checkpoint

The trained model is saved as:

```text
mobilenetv2_aptos_best.pth
```

## 📁 Repository Structure

```text
.
├── data/
│   └── aptos2019/
├── aptos_diabetic_retinopathy.ipynb
├── mobilenetv2_aptos_best.pth
├── README.md
└── requirements.txt
```

> Dataset files and large model checkpoints may be excluded from GitHub depending on their size and distribution restrictions.

## 🛠️ Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Pillow
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

## 🚀 Installation

```bash
git clone https://github.com/<your-username>/aptos-diabetic-retinopathy.git
cd aptos-diabetic-retinopathy
pip install -r requirements.txt
```

## ▶️ Usage

```bash
jupyter notebook aptos_diabetic_retinopathy.ipynb
```

Run the notebook cells to load the dataset, preprocess images, train and progressively fine-tune MobileNetV2, evaluate the model, and generate visualizations.

## 🔬 Future Improvements

- Class-weighted or focal loss for class imbalance
- Better retinal-image preprocessing
- Comparison with EfficientNet, ResNet and ConvNeXt
- Cross-validation
- Hyperparameter optimization
- Improved minority-class performance
- Model calibration and confidence estimation
- External test-set evaluation

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. The model is not a medical diagnostic system and should not be used for clinical decisions.

## 👨‍💻 Author

**Nipun Gupta**

If you found this project useful, consider giving the repository a ⭐.
