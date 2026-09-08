# Brain Tumor Classification using EfficientNet-B0

Deep Learning project for the **classification of brain tumors from MRI images** using **EfficientNet-B0** with PyTorch.

The model classifies MRI images into four categories:

* **Glioma**
* **Meningioma**
* **No Tumor**
* **Pituitary**

The project also includes **data augmentation, model evaluation, confusion matrix analysis, misclassified image analysis, Grad-CAM explainability, and ONNX model export**.

---

## Project Overview

Brain tumor classification from MRI scans is an important computer vision task that can assist medical image analysis.

This project uses **transfer learning with EfficientNet-B0** to automatically classify brain MRI images into four categories.

The pipeline includes:

```text
MRI Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Data Augmentation
     │
     ▼
EfficientNet-B0
     │
     ▼
Model Training
     │
     ▼
Validation
     │
     ▼
Test Evaluation
     │
     ├── Accuracy
     ├── Classification Report
     ├── Confusion Matrix
     └── Misclassified Images
     │
     ▼
Grad-CAM Explainability
     │
     ▼
ONNX Export
```

---

## Model

The project uses **EfficientNet-B0** with transfer learning.

The pretrained EfficientNet-B0 architecture is adapted for the four target classes by replacing its final classification layer.

```python
model = efficientnet_b0(weights="DEFAULT")

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 4)
```

### Why EfficientNet-B0?

EfficientNet provides a good balance between:

* Classification performance
* Computational efficiency
* Model size
* Training speed

EfficientNet-B0 is particularly suitable when the goal is to obtain strong image classification performance without using an excessively large model.

---

## Dataset

The dataset is organized into training and testing directories:

```text
Brain Tumor MRI Dataset/
│
└── classification_task/
    │
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── no_tumor/
    │   └── pituitary/
    │
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── no_tumor/
        └── pituitary/
```

### Classes

| Class        | Description                |
| ------------ | -------------------------- |
| `glioma`     | Glioma tumor               |
| `meningioma` | Meningioma tumor           |
| `no_tumor`   | MRI with no detected tumor |
| `pituitary`  | Pituitary tumor            |

The notebook automatically discovers the images from the class directories and assigns numerical labels.

---

## Data Preprocessing

Images are converted to RGB and resized to **224 × 224 pixels**, matching the expected input resolution of EfficientNet-B0.

### Training Augmentation

The training pipeline applies:

* Resize
* Random resized crop
* Random horizontal flip
* Random rotation
* Color jitter
* Tensor conversion
* ImageNet normalization

```text
Resize
  ↓
RandomResizedCrop
  ↓
RandomHorizontalFlip
  ↓
RandomRotation
  ↓
ColorJitter
  ↓
ToTensor
  ↓
Normalization
```

Validation and test images use deterministic preprocessing without random augmentation.

---

## Train / Validation Split

The original training dataset is divided into:

* **80% training**
* **20% validation**

The split uses stratification to preserve the class distribution.

```python
train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)
```

A fixed `random_state=42` is used to make the split reproducible.

---

## Training Configuration

| Parameter     |             Value |
| ------------- | ----------------: |
| Model         |   EfficientNet-B0 |
| Image Size    |         224 × 224 |
| Batch Size    |                32 |
| Epochs        |                10 |
| Optimizer     |             AdamW |
| Learning Rate |              1e-4 |
| Weight Decay  |              1e-4 |
| Loss Function |  CrossEntropyLoss |
| Scheduler     | ReduceLROnPlateau |

The learning rate scheduler reduces the learning rate when validation loss stops improving.

---

## Training Strategy

The best model is selected according to **validation loss**.

During training, the model tracks:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

The model checkpoint is saved whenever a new best validation loss is achieved.

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), MODEL_PATH)
```

The notebook also automatically loads an existing trained model instead of retraining it when the checkpoint is already available.

---

## Evaluation

After training, the model is evaluated on the independent test set.

The evaluation includes:

### Test Accuracy

```python
accuracy_score(all_labels, all_predictions)
```

### Classification Report

The following metrics are calculated for every class:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

A confusion matrix is generated to visualize correct and incorrect predictions for each tumor class.

---

## Model Performance

The trained model achieved:

### **98.40% Test Accuracy**

| Class                | Precision |  Recall |   F1-Score |
| -------------------- | --------: | ------: | ---------: |
| Glioma               |    98.80% |  97.64% |     98.22% |
| Meningioma           |    96.50% |  99.02% |     97.74% |
| No Tumor             |    98.59% | 100.00% |     99.29% |
| Pituitary            |   100.00% |  97.67% |     98.82% |
| **Overall Accuracy** |           |         | **98.40%** |

The test evaluation and classification report are generated directly by the notebook.

> **Note:** These results are experimental and should not be interpreted as clinical performance. This project is intended for research and educational purposes.

---

## Misclassified Images

The project identifies incorrectly classified test images and displays examples of the errors.

This makes it possible to investigate cases such as:

```text
True Class → Glioma
Predicted → Meningioma
```

The notebook extracts the incorrectly classified samples and visualizes up to ten examples.

This analysis can help identify:

* Similar visual patterns between tumor types
* Difficult MRI cases
* Potential dataset problems
* Model weaknesses

---

# Explainability with Grad-CAM

A major component of this project is **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM provides a visual explanation of the regions that contributed most to the model's prediction.

The target layer used in the project is the final feature layer of EfficientNet-B0:

```python
target_layer = model.features[-1]
cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)
```

The generated visualization contains three views:

```text
┌─────────────────┬─────────────────┬─────────────────┐
│   Original MRI  │    Grad-CAM     │   Overlay       │
│                 │                 │                 │
│     MRI scan    │ Activation Map  │ MRI + Heatmap   │
└─────────────────┴─────────────────┴─────────────────┘
```

Grad-CAM is applied to both:

* Correctly classified images
* Misclassified images

This provides additional insight into **where the model is looking when making its prediction**.

---

## ONNX Export

The trained EfficientNet-B0 model can be exported to **ONNX** for easier deployment and interoperability with different inference environments.

The exported model is:

```text
efficientnet_b0_production.onnx
```

The model expects an input tensor with the shape:

```text
[1, 3, 224, 224]
```

The notebook uses ONNX opset version 15 and performs a model validation check after export.

This makes the model suitable for potential integration into:

* FastAPI applications
* Web applications
* Mobile applications
* Cloud inference services
* Other ONNX-compatible environments

---

## Technologies

### Deep Learning

* Python
* PyTorch
* TorchVision
* EfficientNet-B0
* Transfer Learning

### Computer Vision

* Pillow
* NumPy
* TorchVision Transforms

### Machine Learning

* Scikit-learn
* Classification Report
* Confusion Matrix
* Train/Validation Split

### Explainable AI

* Grad-CAM
* PyTorch Grad-CAM

### Visualization

* Matplotlib
* Seaborn

### Deployment

* ONNX

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-classification.git
cd brain-tumor-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install torch torchvision
pip install pillow numpy matplotlib seaborn
pip install scikit-learn
pip install grad-cam
pip install onnx
```

---

## Dataset Setup

Download and place the dataset in the following structure:

```text
Brain Tumor MRI Dataset/
└── classification_task/
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── no_tumor/
    │   └── pituitary/
    │
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── no_tumor/
        └── pituitary/
```

Update the dataset path in the notebook:

```python
DATASET_PATH = Path("path/to/Brain Tumor MRI Dataset")
```

---

## Running the Project

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then run:

```text
Tumor.ipynb
```

The notebook performs the complete pipeline:

1. Load dataset
2. Preprocess MRI images
3. Apply data augmentation
4. Split training and validation data
5. Load EfficientNet-B0
6. Train the classifier
7. Save the best model
8. Evaluate on the test set
9. Generate the confusion matrix
10. Analyze misclassified images
11. Generate Grad-CAM visualizations
12. Export the model to ONNX

---

## Project Structure

```text
brain-tumor-classification/
│
├── classification_task/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   │
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── no_tumor/
│       └── pituitary/
│
├── Tumor.ipynb
├── best_efficientnet_b0.pth
├── efficientnet_b0_production.onnx
│
├── Training vs Validation Loss.png
├── Training vs Validation Accuracy.png
├── confusion_matrix.png
│
└── README.md
```

> The MRI dataset itself should generally **not be committed to GitHub**. Add it to `.gitignore` or provide instructions for obtaining it separately.


---

## Important Disclaimer

This project is intended **for educational and research purposes only**.

The model is **not a medical diagnostic system** and should not be used to diagnose, treat, or make clinical decisions about patients.

A high test accuracy on a specific dataset does not guarantee reliable performance on clinical MRI scans from different hospitals, scanners, populations, or acquisition protocols.

---

## Author

**Amine Rafrafi**

Software Engineering Graduate | Data Science & AI Enthusiast

Interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Data Science
* Computer Vision
* MLOps

---

## License

This project is intended for educational and research purposes.

Please check the original dataset's license and terms of use before redistributing or using the dataset.
