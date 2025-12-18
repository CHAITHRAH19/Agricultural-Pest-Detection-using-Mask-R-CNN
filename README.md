# 🌾 Agricultural Pest Detection using Mask R-CNN

This project implements a **Mask R-CNN–based deep learning framework** for **large-scale agricultural pest detection and instance segmentation**, with a special focus on **tiny object detection**. The system is evaluated on the **IP102 pest dataset**, addressing real-world challenges such as **small object size, watermark noise, and class imbalance**.

---

## 📌 Project Overview

Early pest detection is essential for improving crop yield and minimizing pesticide use. Traditional manual inspection is time-consuming and error-prone. This project applies **computer vision and deep learning** techniques to automatically detect and segment pests in agricultural images.

**Key contributions:**

* Custom Mask R-CNN architecture for pest segmentation
* Handling tiny objects using FPN and anchor refinement
* Data augmentation for improved generalization
* Large-scale benchmarking on IP102 dataset

---

## 🗂 Dataset

* **Dataset**: IP102 – Large-scale agricultural pest dataset
* **Source**:
  [https://universe.roboflow.com/pest-segmentations/paddy-pests-segmentations](https://universe.roboflow.com/pest-segmentations/paddy-pests-segmentations)
* **Images**:

  * Training: 45,095
  * Validation: 7,518
* **Challenges**:

  * Very small pest objects
  * Watermarks in images
  * Severe class imbalance

---

## 🛠 Technologies Used

* **Language**: Python
* **Frameworks**:

  * PyTorch
  * Torchvision
* **Model**: Mask R-CNN
* **Backbone**: ResNet-50 + Feature Pyramid Network (FPN)
* **Optimization**:

  * AdamW optimizer
  * Learning rate scheduling
* **Preprocessing**:

  * Image resizing (512 × 512)
  * Normalization (ImageNet)
  * Data augmentation

---

## 🔄 Methodology

```
Dataset Collection
        ↓
Preprocessing & Augmentation
        ↓
Mask R-CNN Customization
        ↓
Training & Fine-Tuning
        ↓
Evaluation & Visualization
```

---

## 🔁 Data Augmentation

To improve robustness and reduce overfitting:

* Horizontal & vertical flips
* Brightness and contrast adjustment
* Image resizing and normalization

---

## 🧠 Model Architecture

* **Architecture**: Mask R-CNN
* **Backbone**: ResNet-50 (pre-trained on COCO)
* **Key Features**:

  * Feature Pyramid Network for multi-scale detection
  * Custom prediction heads for pest classes
  * Non-Maximum Suppression (NMS) for duplicate removal

---

## ⚙️ Training Details

* **Optimizer**: AdamW
* **Learning Rate**: `1e-5`
* **Loss Components**:

  * Classification loss
  * Bounding box regression loss
  * Mask loss
* **Post-processing**:

  * Confidence threshold = 0.5
  * NMS applied to reduce false positives

---

## 📊 Evaluation Results

| Metric    | Value      |
| --------- | ---------- |
| Recall    | **88.60%** |
| IoU       | **81.00%** |
| Precision | **31.62%** |

### Observations

* High recall and localization accuracy
* Precision affected by class imbalance and watermark noise
* Strong performance on majority classes
* Difficulty detecting very small or minority-class pests

---

## 🖼 Visual Results

* Correct segmentation in dense pest regions
* False positives caused by watermark artefacts
* Improved detection using multi-scale feature extraction

---

## 📁 Repository Structure

```
├── data/
│   ├── train/
│   ├── val/
├── model/
├── training/
├── evaluation/
├── utils/
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/your-username/agricultural-pest-detection.git
cd agricultural-pest-detection

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Evaluate model
python evaluate.py
```

---

## ⚠️ Limitations

* Watermarks introduce noise
* Class imbalance reduces minority class precision
* Tiny pest size limits feature extraction
* False positives remain a challenge

---

## 🔮 Future Work

* Watermark removal using inpainting
* Anchor box refinement for tiny pests
* Focal loss for hard examples
* Oversampling & GAN-based data augmentation
* Higher-resolution input images

---

