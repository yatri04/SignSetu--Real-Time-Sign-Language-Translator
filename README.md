# 🤟 SignSetu - Real-Time Sign Language Translator
**SGP Project - Proposed System Architecture and Development Plan**
>
> SignSetu is a planned real-time computer vision and deep learning system that aims to recognize hand signs from a live camera feed and convert them into readable text.

---

## 📌 Project Status

| Component | Current Status |
|---|---|
| Problem identification | ✅ Completed |
| Project concept | ✅ Decided |
| Dataset selection | ✅ Planned / selected |
| System architecture | ✅ Designed |
| Model architecture | ✅ Proposed |
| Data preprocessing pipeline | ⏳ To be implemented |
| Model training | ⏳ Not started |
| Model evaluation | ⏳ Not started |
| Real-time webcam integration | ⏳ Not started |
| User interface | ⏳ To be implemented |
| Final deployment | ⏳ Future phase |

# 1. 🎯 Project Overview

SignSetu is proposed as a **real-time sign language recognition and translation system** using computer vision and deep learning.

The basic idea is:

**Camera → Hand Detection → Hand Region Extraction → Image Preprocessing → Deep Learning Model → Sign Classification → Text Output**

The system will receive frames from a webcam, identify the user's hand/sign, extract the relevant hand region, and classify the observed sign into one of the classes learned by the model.

The initial version will focus on **static hand signs** rather than continuous dynamic sentences. This allows the project to establish a reliable image-classification pipeline before extending the system to temporal/dynamic sign recognition.

---

# 2. 💡 Problem Statement

Communication between sign-language users and people who do not understand sign language can be difficult because conventional communication systems are primarily designed around spoken or written language.

Existing sign-language recognition approaches may work well under controlled conditions but can face challenges such as:

- Different backgrounds
- Different lighting conditions
- Different hand positions and orientations
- Variation between individual signers
- Real-time processing requirements
- Visually similar hand signs

SignSetu aims to explore how **deep learning + real-time hand detection** can be combined to create a practical sign recognition pipeline.

---

# 3. 🎯 Objectives

The major objectives of SignSetu are:

1. Build a real-time sign language recognition system.
2. Use computer vision to detect and isolate the hand from a webcam frame.
3. Train a deep learning classification model on a sign-language dataset.
4. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
5. Integrate the trained model with a live webcam.
6. Display the predicted sign and confidence score in real time.
7. Create a foundation for future word, sentence, and speech-level translation.

---

# 4. 🧠 Core Idea

The system will be divided into two major stages.

### Stage 1 - Model Development

The model will learn the relationship between hand-sign images and their corresponding labels.

```text
Dataset
   ↓
Data Cleaning
   ↓
Train / Validation / Test Split
   ↓
Data Augmentation
   ↓
Image Preprocessing
   ↓
Deep Learning Model
   ↓
Training
   ↓
Validation
   ↓
Final Evaluation
```

### Stage 2 - Real-Time Recognition

The trained model will then be connected to a webcam.

```text
Webcam
   ↓
Video Frame
   ↓
Hand Detection
   ↓
Hand Region / Landmark Extraction
   ↓
Resize + Normalize
   ↓
Trained Deep Learning Model
   ↓
Class Prediction
   ↓
Confidence Score
   ↓
Text Display
```

---

# 5. 📊 Dataset

## Initial Dataset Decision

For this project the considered dataset is **ASL Alphabet Dataset** from Kaggle.
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
The dataset contains:

- Approximately **87,000 images**
- **26 alphabet classes (A–Z)**
- RGB images
- Static hand-sign images
- Controlled/relatively plain backgrounds


---

# 6. 🔄 Data Preprocessing

Before training, the images will pass through a preprocessing pipeline.

### Planned preprocessing steps

1. Load images.
2. Verify image labels and class distribution.
3. Remove corrupted or unusable samples.
4. Resize images to the model input size.
5. Normalize pixel values.
6. Apply data augmentation to the training set.
7. Create PyTorch DataLoaders.

### Proposed augmentation

Depending on dataset characteristics, the following transformations will be evaluated:

- Small random rotations
- Translation
- Scaling
- Brightness variation
- Contrast variation
- Horizontal flipping where semantically valid
- Normalization

> Augmentation will be applied carefully because some sign classes can change meaning when transformed incorrectly.

---

# 7. 🤖 Proposed Deep Learning Approach

The primary proposed approach is **Transfer Learning using ResNet18**.

ResNet18 is a convolutional neural network (CNN) architecture containing residual connections. Instead of training the entire network from random initialization, we plan to start from a model pretrained on ImageNet and adapt its final classification layer to our sign classes.

### Why ResNet18?

- Relatively lightweight
- Good image-feature extraction capability
- Suitable for transfer learning
- Faster than many larger CNN architectures
- Practical for real-time inference
- Well supported by PyTorch/Torchvision

---

# 8. 🏗️ Proposed Model Architecture

```text
Input Image
224 × 224 × 3
       │
       ▼
ResNet18 Backbone
Pretrained on ImageNet
       │
       ├── Convolutional Layer
       ├── Residual Block
       ├── Residual Block
       ├── Residual Block
       └── Residual Block
       │
       ▼
Global Average Pooling
       │
       ▼
Feature Vector
       │
       ▼
Custom Classification Head
       │
       ├── Dropout
       ├── Fully Connected Layer
       ├── ReLU
       ├── Dropout
       └── Fully Connected Layer
       │
       ▼
Sign Classes
       │
       ▼
Softmax / Class Probabilities
       │
       ▼
Predicted Sign
```

For an initial 26-class alphabet experiment, the final classification layer will contain **26 output classes**.

---

# 9. 🔍 Why Transfer Learning?

Training a deep CNN completely from scratch requires a large amount of data and computational resources.

With transfer learning:

```text
ImageNet Pretrained Model
          ↓
General Visual Features
          ↓
Fine-Tuning
          ↓
Sign Language Classification
```

The pretrained network already contains useful low-level and mid-level visual representations such as edges, textures, shapes, and patterns.

We will adapt these learned features to the sign-language classification task.

---

# 10. 🖐️ Hand Detection and Isolation

A major challenge in real-time recognition is **domain shift**.

For example, a model trained on images with relatively simple backgrounds may perform poorly when a webcam introduces:

- Room backgrounds
- Shadows
- Different lighting
- Objects behind the hand
- Different camera distances

To reduce this problem, SignSetu plans to use **MediaPipe Hand Landmarker / hand detection** before classification.


### Proposed pipeline

```text
Webcam Frame
      ↓
MediaPipe Hand Detection
      ↓
Hand Bounding Region / Landmarks
      ↓
Crop / Isolate Hand
      ↓
Resize
      ↓
Normalize
      ↓
ResNet18
      ↓
Sign Prediction
```

The hand-detection stage is therefore not the classifier itself. It acts as a **computer-vision preprocessing stage** that helps the deep-learning model receive a cleaner input.

---

# 11. 🏗️ Complete System Architecture

```text
                         SIGNSETU
                            │
                            ▼
                    ┌───────────────┐
                    │ Webcam Input  │
                    └───────┬───────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Frame Acquisition  │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ MediaPipe Hand     │
                  │ Detection           │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Hand ROI /         │
                  │ Landmark Extraction│
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Preprocessing      │
                  │ Resize + Normalize │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ ResNet18 CNN       │
                  │ Transfer Learning  │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Classification     │
                  │ A-Z / Sign Classes │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Confidence Score   │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Real-Time UI       │
                  │ Predicted Text     │
                  └────────────────────┘
```

---

# 12. 🧩 Major Components

## Component 1 — Dataset Layer

Responsible for:

- Dataset loading
- Class mapping
- Data validation
- Train/validation/test splitting
- Augmentation

---

## Component 2 — Computer Vision Layer

Responsible for:

- Webcam frame acquisition
- Hand detection
- Hand localization
- Hand-region extraction
- Frame preprocessing

**Technology:** OpenCV + MediaPipe

---

## Component 3 — Deep Learning Layer

Responsible for:

- Feature extraction
- Sign classification
- Confidence estimation

**Technology:** PyTorch + Torchvision

**Proposed model:** ResNet18 with transfer learning

---

## Component 4 — Evaluation Layer

We will evaluate the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance

---

## Component 5 — Real-Time Inference Layer

Responsible for:

- Loading the trained model
- Receiving webcam frames
- Running hand detection
- Performing inference
- Displaying predictions
- Measuring FPS and latency

---

# 16. 🧪 Experimental Plan

The project will be implemented incrementally.

### Phase 1 - Dataset Analysis

- Download/prepare the selected dataset.
- Explore class distribution.
- Display sample images.
- Check image dimensions and quality.
- Identify potential dataset imbalance.

### Phase 2 - Preprocessing

- Implement train/validation/test split.
- Implement resizing and normalization.
- Test augmentation strategies.
- Create DataLoaders.

### Phase 3 - Baseline Model

- Implement a basic CNN or initial ResNet18 configuration.
- Train on the dataset.
- Record training/validation loss.
- Record accuracy.

### Phase 4 - Transfer Learning

- Load pretrained ResNet18.
- Replace classification head.
- Fine-tune on sign-language data.
- Compare performance with the baseline.

### Phase 5 - Model Evaluation

Generate:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training/validation curves
- Sample predictions

### Phase 6 - Real-Time Integration

- Connect webcam using OpenCV.
- Detect hand using MediaPipe.
- Extract hand region.
- Pass processed image to the trained model.
- Display predicted sign.

### Phase 7 - Optimization

- Reduce inference latency.
- Improve preprocessing.
- Test different lighting/background conditions.
- Evaluate generalization to unseen users.

### Phase 8 - Final System

Integrate:

```text
Webcam
+
MediaPipe
+
Deep Learning Model
+
Prediction Engine
+
User Interface
```

# 13. 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | ResNet18 and image transformations |
| OpenCV | Webcam and image processing |
| MediaPipe | Real-time hand detection and landmarks |
| NumPy | Numerical operations |
| Pandas | Dataset analysis |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation metrics |
| Jupyter Notebook | Experimentation and model development |

---

# 14. 🔬 Deep Learning Workflow

The planned training workflow is:

```text
Dataset
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
DataLoader
   ↓
Pretrained ResNet18
   ↓
Replace Final Layer
   ↓
Forward Pass
   ↓
Cross Entropy Loss
   ↓
Backpropagation
   ↓
Optimizer Update
   ↓
Validation
   ↓
Checkpoint Best Model
   ↓
Final Test Evaluation
```

### Initial training configuration

The exact values will be determined experimentally, but the initial plan is to investigate:

- Optimizer: Adam
- Loss: CrossEntropyLoss
- Batch size: approximately 32
- Learning rate: approximately 0.001 initially
- Learning-rate scheduling
- Early stopping
- Weight decay
- Dropout

These are **planned starting points**, not final experimental results.

---
# 15. 🧭 Development Roadmap

```text
                 SIGNSETU ROADMAP

          ┌───────────────────────┐
          │ Problem Definition    │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Dataset Preparation   │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Data Preprocessing    │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Baseline CNN          │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ ResNet18 Transfer     │
          │ Learning              │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Model Evaluation      │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ MediaPipe Integration │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Real-Time Webcam      │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Sign → Text           │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Optimization &        │
          │ Testing               │
          └───────────┬───────────┘
                      ↓
          ┌───────────────────────┐
          │ Final Demonstration   │
          └───────────────────────┘
```

# 16. 📌 Expected Final Outcome

The expected first-version output is:

```text
User performs sign
        ↓
Camera captures sign
        ↓
Hand is detected
        ↓
Hand region is processed
        ↓
Deep learning model predicts class
        ↓
Confidence score generated
        ↓
Predicted character/sign displayed
```

# 🏁 Conclusion

SignSetu is planned as a real-time sign-language recognition system combining **computer vision, hand tracking, transfer learning, and deep learning**.

The first milestone is intentionally focused on **static sign recognition** so that the complete pipeline can be established and evaluated:

**Dataset → Preprocessing → CNN/Transfer Learning → Evaluation → Hand Detection → Real-Time Inference**

After achieving a reliable first version, the system can be expanded toward dynamic signs, word formation, sentence-level translation, and text-to-speech.
