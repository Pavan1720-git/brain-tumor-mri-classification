# 🧠 Brain Tumor MRI Image Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)

An end-to-end deep learning system for classifying brain MRI scans into four categories using Custom CNN and Transfer Learning models (MobileNetV2, EfficientNetB0).

> ⚕️ **Disclaimer**: This project is for **educational and research purposes only** and is not a medical diagnostic tool.

---

## 📋 Problem Statement

Brain tumors are life-threatening conditions that require early detection for better prognosis. Manual analysis of MRI scans is time-consuming and subject to human error. This project demonstrates how deep learning can assist in automating brain tumor classification from MRI images into four categories:

| Class | Description |
|-------|-------------|
| **Glioma** | Tumor arising from glial cells; can be benign or malignant |
| **Meningioma** | Tumor of the meninges; often slow-growing |
| **No Tumor** | No tumor detected in the MRI scan |
| **Pituitary** | Tumor in the pituitary gland at the base of the brain |

---

## 📊 Dataset

The dataset was provided for the internship and is **not included in this repository**.

| Split | Total | Glioma | Meningioma | No Tumor | Pituitary |
|-------|-------|--------|------------|----------|-----------|
| **Train** | 1,695 | 564 | 358 | 335 | 438 |
| **Valid** | 502 | 161 | 124 | 99 | 118 |
| **Test** | 246 | 80 | 63 | 49 | 54 |
| **Total** | **2,443** | **805** | **545** | **483** | **610** |

- **Format**: RGB JPG, 640×640 pixels (resized to 224×224 for training)
- **Labels**: One-hot encoded in `_classes.csv` per split; also organized in class-named subfolders
- **Splits**: Pre-split — used as-is (no re-splitting)

> To use the dataset, place the extracted folders as:
> ```
> data/train/glioma/     data/train/meningioma/
> data/train/no_tumor/   data/train/pituitary/
> data/valid/ ...        data/test/ ...
> ```

---

## 🛠️ Technologies

- **Python 3.11**
- **TensorFlow / Keras** — model building and training
- **Custom CNN** — built from scratch for this dataset
- **Transfer Learning** — MobileNetV2, EfficientNetB0
- **Scikit-learn** — metrics (precision, recall, F1-score, confusion matrix)
- **Matplotlib / Seaborn** — visualisations
- **Streamlit** — web application for predictions
- **Pillow** — image loading and preprocessing

---

## 📁 Project Structure

```
brain-tumor-mri-classification/
│
├── data/                          # Dataset (not committed to Git)
│   ├── train/                     # 1,695 training images
│   ├── valid/                     # 502 validation images
│   └── test/                      # 246 test images
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Custom_CNN.ipynb        # Custom CNN training
│   ├── 03_Transfer_Learning.ipynb # MobileNetV2 + EfficientNetB0
│   └── 04_Model_Evaluation.ipynb  # Final comparison & selection
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # All hyperparameters & paths
│   ├── data_utils.py              # Dataset inspection utilities
│   ├── preprocessing.py           # Data generators & preprocessing
│   ├── models.py                  # Model factory + callbacks
│   ├── evaluate.py                # Metrics, plots, reports
│   └── train.py                   # End-to-end training pipeline
│
├── models/                        # Saved model weights
│   ├── custom_cnn_best.keras
│   ├── mobilenetv2_best.keras
│   ├── efficientnetb0_best.keras
│   ├── best_brain_tumor_model.keras  ← deployment model
│   ├── class_names.json
│   └── deployment_meta.json
│
├── results/                       # Generated plots and reports
│   ├── dataset_distribution.png
│   ├── sample_images.png
│   ├── augmentation_preview.png
│   ├── custom_cnn_accuracy.png
│   ├── custom_cnn_loss.png
│   ├── custom_cnn_confusion_matrix.png
│   ├── mobilenetv2_accuracy.png
│   ├── mobilenetv2_loss.png
│   ├── mobilenetv2_confusion_matrix.png
│   ├── efficientnetb0_accuracy.png
│   ├── efficientnetb0_loss.png
│   ├── efficientnetb0_confusion_matrix.png
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── classification_reports/
│   └── misclassified_samples/
│
├── app.py                         # Streamlit web application
├── requirements.txt
├── generate_notebooks.py          # Notebook generation script
├── .gitignore
└── README.md
```

---

## 🔄 Project Workflow

```
Dataset (2,443 MRI images)
        ↓
    Data Inspection
    (class counts, CSV validation, corrupt check)
        ↓
    EDA (Notebook 01)
    (distributions, sample images, quality checks)
        ↓
    Preprocessing
    (resize 224×224, RGB, architecture-specific normalization)
        ↓
    Data Augmentation (train only)
    (rotation ±10°, shift ±5%, zoom ±10%, brightness ±10%)
        ↓
    Model Training (Notebooks 02 + 03)
    ┌──────────────┬──────────────────┬─────────────────┐
    │  Custom CNN  │  MobileNetV2     │  EfficientNetB0 │
    │  (scratch)   │  (frozen→finetune│  (frozen→finetune│
    └──────────────┴──────────────────┴─────────────────┘
        ↓
    Model Evaluation (Notebook 04)
    (accuracy, precision, recall, F1, confusion matrix)
        ↓
    Model Comparison & Selection
        ↓
    Streamlit Application (app.py)
```

---

## 🧠 Models

### Custom CNN
- 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- BatchNormalization + ReLU + MaxPooling in each block
- GlobalAveragePooling → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Softmax(4)
- L2 regularization on dense layers

### MobileNetV2
- Pretrained on ImageNet (frozen base → fine-tune top 30 layers)
- GlobalAveragePooling → Dense(256) → Dropout(0.4) → Softmax(4)
- Phase 1 LR: 1e-3 | Fine-tune LR: 1e-5

### EfficientNetB0
- Pretrained on ImageNet (frozen base → fine-tune top 20 layers)
- GlobalAveragePooling → Dense(256) → Dropout(0.4) → Softmax(4)
- Phase 1 LR: 1e-3 | Fine-tune LR: 1e-5

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 × 3 |
| Batch size | 32 |
| Custom CNN max epochs | 50 |
| Transfer learning frozen epochs | 25 |
| Fine-tuning epochs | 20 |
| Optimizer | Adam |
| Initial LR | 1e-3 |
| Fine-tune LR | 1e-5 |
| Loss | Categorical Cross-Entropy |
| Early stopping patience | 10 |
| ReduceLROnPlateau | factor=0.5, patience=5 |

---

## 📈 Results

The models were evaluated strictly on the **246 test images** (held out, never used during training or tuning). The results below are the **actual measured values** generated by the project pipeline:

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score | Status |
|-------|---------------|-------------------|-----------------|-------------------|--------|
| **MobileNetV2** | **91.06%** (224/246) | **0.9108** | **0.9106** | **0.9092** | 🏆 **Selected for Deployment** |
| **EfficientNetB0** | 89.43% (220/246) | 0.8985 | 0.8943 | 0.8948 | Runner-up |
| **Custom CNN** | 44.31% (109/246) | 0.3603 | 0.4431 | 0.3588 | Baseline (Scratch) |

### Detailed Per-Class Breakdown (Best Model: MobileNetV2)

| Class | Precision | Recall | F1-score | Support (Images) |
|-------|-----------|--------|----------|------------------|
| **Glioma** | 0.9277 | 0.9625 | 0.9448 | 80 |
| **Meningioma** | 0.8644 | 0.8095 | 0.8361 | 63 |
| **No Tumor** | 0.9545 | 0.8571 | 0.9032 | 49 |
| **Pituitary** | 0.9000 | 1.0000 | 0.9474 | 54 |
| **Overall Weighted** | **0.9108** | **0.9106** | **0.9092** | **246** |

### Key Findings & Analysis
- **Transfer Learning Impact**: Pretrained representations from ImageNet drastically outperformed the scratch CNN (+46.75% accuracy), which is typical for domain-specific medical tasks with limited sample sizes (~1,700 training scans).
- **Pituitary Detection**: Both MobileNetV2 and EfficientNetB0 achieved **100% recall (1.0000)** on Pituitary tumor test images (54 out of 54 detected).
- **Model Selection Decision**: MobileNetV2 was chosen for final deployment in the Streamlit app because it achieved the highest F1-score (**0.9092**), highest overall accuracy (**91.06%**), balanced per-class performance, and low inference latency suitable for edge/local deployment.

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate on **Windows**:
```bash
venv\Scripts\activate
```

Activate on **macOS/Linux**:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Place the dataset folders as:
```
data/
├── train/glioma/       (564 images)
├── train/meningioma/   (358 images)
├── train/no_tumor/     (335 images)
├── train/pituitary/    (438 images)
├── valid/ ...
└── test/  ...
```

---

## 🏃 Running the Project

### Option A — Full training pipeline (recommended)

```bash
python -m src.train
```

This trains all three models sequentially and saves everything automatically.

### Option B — Run notebooks step by step

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Custom_CNN.ipynb
jupyter notebook notebooks/03_Transfer_Learning.ipynb
jupyter notebook notebooks/04_Model_Evaluation.ipynb
```

### Option C — Run the Streamlit app (after training)

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Streamlit Application

The web app allows you to:
1. Upload any brain MRI image (JPG/JPEG/PNG)
2. View predicted class with confidence score
3. See probability distribution across all 4 classes
4. Read class descriptions

---

## ⚠️ Known Limitations

- **Small dataset**: 2,443 images is small for deep learning; transfer learning compensates significantly.
- **Class imbalance**: Glioma is over-represented; class weights are applied during training.
- **No external validation**: The model has only been evaluated on the provided test set.
- **Educational prototype**: This is NOT a medical device. Do not use for actual diagnosis.
- **CPU training**: Without GPU, training may take several hours.

---

## 📜 License

This project is created for internship/educational purposes.

---

## ⚕️ Medical Disclaimer

This application is intended for **educational and research purposes only**. It is **not a medical diagnostic tool** and should **not be used as a substitute** for evaluation by a qualified healthcare professional. Confidence scores do not represent medical certainty.
