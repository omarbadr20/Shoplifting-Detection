 # Shoplifting Detection

This repository contains a PyTorch implementation of a 3D CNN-based spatial-temporal video classification model designed to distinguish between **normal shopping behavior** and **shoplifting/theft acts** in retail surveillance footage. 

Our pipeline is optimized to handle variable video lengths, apply spatially consistent data augmentations to prevent domain overfitting, and evaluate performance using standard classification metrics.

---

## Table of Contents
1. [Objective & Tasks](#objective--tasks)
2. [Dataset Overview](#dataset-overview)
3. [System Architecture & Key Implementations](#system-architecture--key-implementations)
   * [Uniform Frame Sampling](#1-uniform-frame-sampling)
   * [Spatially Consistent Augmentations](#2-spatially-consistent-augmentations)
   * [3D CNN Model](#3-3d-cnn-model)
4. [Training and Generalization Experiments](#training-and-generalization-experiments)
   * [Experiment 1: The Shortcut Learning Challenge](#experiment-1-the-shortcut-learning-challenge)
   * [Experiment 2: Mitigating Domain Overfitting](#experiment-2-mitigating-domain-overfitting)
5. [Installation & Requirements](#installation--requirements)
6. [Usage Instructions](#usage-instructions)
   * [1. Dataset Exploration](#1-dataset-exploration)
   * [2. Training the Pipeline](#2-training-the-pipeline)
   * [3. Single Video Inference](#3-single-video-inference)

---

## Objective & Tasks

The goal of this project is to build a robust video classification model that operates on variable-length security camera clips. The key project requirements completed in this repository are:
* **Exploratory Data Analysis:** Profile duration, frame counts, and resolution across the dataset.
* **Uniform Frame Sampling:** Sample a fixed number of frames ($T$) spread evenly across any video duration, resolving variable sequence lengths.
* **Spatially Consistent Augmentation:** Apply identical random spatial augmentations (crop, flip, color jitter, grayscale, cutout) to all frames within a given video clip to maintain temporal consistency.
* **Dataset Pipeline:** Build a PyTorch `Dataset` that yields structured video tensors of shape `(T, C, H, W)`.
* **Model Implementation:** Develop a 3D Convolutional Neural Network (3D CNN) to capture joint spatial and temporal cues.
* **Robust Evaluation:** Train and track accuracy, precision, recall, and F1-score with continuous progress visualization.

---

## Dataset Overview

The dataset consists of **855 retail surveillance video clips** split into two folders:
* **shop lifters (Theft):** 324 videos
* **non shop lifters (Normal):** 531 videos

### Dataset Profile
An analysis of the dataset revealed the following metadata statistics:
* **Resolution:** $704 \times 576$ uniform resolution across all 855 clips.
* **Video Duration:** Varies from a minimum of **3.0 seconds** to a maximum of **74.0 seconds**.
* **Frame Counts:** Varies from a minimum of **75 frames** to a maximum of **1850 frames**.
* **Class Imbalance:** Mild imbalance present (~62% normal vs. ~38% shoplifting).

---

## System Architecture & Key Implementations

### 1. Uniform Frame Sampling
To ingest variable-length video clips into the 3D CNN, we define a target frame count $T = 16$. The sampling strategy computes evenly spaced frame indices using `np.linspace` spanning the entire range $[0, N-1]$ (where $N$ is the total frames in the clip). For shorter clips, a fallback padding duplicates the final frame. This ensures that the sampled frames evenly cover the full duration of any video.

### 2. Spatially Consistent Augmentations
Standard image augmentations applied independently to each video frame disrupt temporal continuity. In our `ShopliftingDataset` class, spatial augmentation parameters are randomly selected **once per video clip** and applied identically across all $T$ sampled frames. 

To prevent the network from memorizing static store backgrounds (shortcut learning), we implemented several robust spatial augmentations:
* **Consistent Random Crop:** Randomly crops a $112 \times 112$ patch from a resized $128 \times 128$ sequence.
* **Widened Color Jitter & Hue Shift:** Color adjustments to prevent reliance on specific store layouts or clothing colors.
* **Random Grayscale (20% probability):** Desaturates the clip to force learning of shapes, outlines, and movement rather than color indicators.
* **Spatial Cutout / Erasing (30% probability):** Randomly blocks out an identical region across all frames. This forces the model to ignore specific static elements (e.g., cash registers, specific shelves) and focus on human motion.

### 3. 3D CNN Model
The model architecture uses 3D convolutions (`nn.Conv3d`) to extract spatial and temporal features simultaneously from full clip tensors. 

```
Input (B, T, C, H, W) 
  --> Permuted to (B, C, T, H, W)
  --> Block 1: Conv3D (3->16) + BatchNorm3D + ReLU + MaxPool3D
  --> Block 2: Conv3D (16->32) + BatchNorm3D + ReLU + MaxPool3D
  --> Block 3: Conv3D (32->64) + BatchNorm3D + ReLU + MaxPool3D
  --> Block 4: Conv3D (64->128) + BatchNorm3D + ReLU + AdaptiveAvgPool3D
  --> Fully Connected Classifier with Dropout (p=0.5) 
  --> Output (B, 2)
```

---

## Training and Generalization Experiments

We conducted two core training runs to observe generalization capabilities.

### Experiment 1: The Shortcut Learning Challenge
In the initial run, the model was trained for **150 epochs** using standard scaling and center-cropping (minimal noise).
* **Result:** The model achieved a **100% Accuracy and F1-score** on its validation and test splits.
* **Failure Mode (Domain Shift):** When evaluated on a completely different external dataset, the model failed, predicting "theft" for almost all inputs.
* **Analysis:** The network fell victim to *Shortcut Learning*. It memorized static environmental structures unique to the training camera setups (specific shelves, background colors) rather than generalized human activities.

### Experiment 2: Mitigating Domain Overfitting
We introduced the enhanced **Spatially Consistent Augmentation** pipeline (with random grayscaling and spatial cutout) and trained for **20 epochs**.

| Metric | Training Progression (Epoch 20) | Final Holdout Test Set Performance |
| :--- | :--- | :--- |
| **Loss** | Decreased smoothly (~0.696 $\rightarrow$ ~0.675) | **0.6568** |
| **Accuracy**| Increased steadily (~54.0% $\rightarrow$ ~61.5%) | **68.99%** |
| **F1-Score**| Improved steadily (~0.330 $\rightarrow$ ~0.470) | **50.00%** |

* **Analysis:** Because static shortcuts were heavily disrupted, the network learned at a slower, more realistic pace. Overfitting was highly mitigated, and the validation loss tracked the training loss downwards without the erratic fluctuations seen in Experiment 1. This model relies on motion-based dynamics, making it far more robust to domain shifts.

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. You can install all necessary dependencies using the following commands:

```bash
pip install torch torchvision opencv-python scikit-learn pandas matplotlib tqdm pillow
```

---

## Usage Instructions

### 1. Dataset Exploration
Run the dataset exploration script to print descriptive statistics about your video formats, durations, frame rates, and folder structures:

```python
from main import explore_dataset

explore_dataset("./shoplifting_dataset")
```

### 2. Training the Pipeline
The primary orchestrator splits your data, runs training with progress bars, and automatically tracks training vs. validation metrics. It saves the model weights to `best_video_classifier.pth` when validation loss decreases, and plots the learning history.

To run the pipeline:
```bash
python main.py
```

### 3. Single Video Inference
To run inference on an individual video file using the best-performing saved checkpoint, run `test_inference.py`:

1. Edit the target video path inside `test_inference.py`:
   ```python
   target_video = "./shoplifting_dataset/shop lifters/example_clip.mp4"
   ```
2. Execute the script:
   ```bash
   python test_inference.py
   ```

Outputs will display as follows:
```
==========================================
           SINGLE FEED INFERENCE RESULTS   
==========================================
Target Video Path : ./shoplifting_dataset/shop lifters/example_clip.mp4
Prediction Output : Shoplifting/Theft Act
Confidence Level  : 84.12%
Raw Probabilities :
  - Normal Behavior : 0.1588
  - Theft Behavior  : 0.8412
==========================================
```
