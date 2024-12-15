
# LANCET: Lightweight Attention-enhanced CNN-based Emotion Recognition Network

This repository contains the official implementation of **LANCET**, a novel lightweight neural network architecture designed for robust speech emotion recognition in clean and noisy conditions. 

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Dataset Preparation](#dataset-preparation)  
   - [Download IEMOCAP Dataset](#download-iemocap-dataset)  
   - [Augment IEMOCAP Dataset](#augment-iemocap-dataset)  
3. [Feature Extraction](#feature-extraction)  
4. [Model Implementation](#model-implementation)  
5. [Training and Evaluation](#training-and-evaluation)  
6. [Citation](#citation)  
7. [Contact](#contact)  

---

## Introduction

LANCET introduces a **Multiscale Channel-wise Attention Mechanism**, a **Temporal Convolutional Network (TCN)**, and **Frame-wise Attention** to effectively capture spectral and temporal dependencies in speech signals. The architecture is designed to perform well in both clean and challenging acoustic environments, leveraging the IEMOCAP dataset for training and evaluation.  
This repository also includes scripts for dataset preparation, feature extraction, and training/testing of LANCET and competing methods.

---

## Dataset Preparation

### Download IEMOCAP Dataset

1. **Download the IEMOCAP Dataset**:  
   The IEMOCAP dataset is publicly available for academic use. You can request access and download it from the [official website](https://sail.usc.edu/iemocap/iemocap_release.htm).  
   
2. **Organize the Dataset**:  
   Once downloaded, place the dataset in the following directory:  
   ```
   datasets/IEMOCAP/
   ```

---

### Augment IEMOCAP Dataset

To evaluate performance under noisy conditions, augment the IEMOCAP dataset with noise types (e.g., babble, music, ambient noise) at various signal-to-noise ratio (SNR) levels.

1. **Run the Augmentation Script**:  
   Use the script provided in the repository to augment the dataset:  
   ```
   python datasets/augment_IEMOCAP.py
   ```
   - This script explains how to add different noise types and SNR levels to the original dataset.  
   - Augmented audio files will be saved in the `datasets/IEMOCAP_augmented/` folder.  

2. **Rename Audio Files**:  
   After augmentation, standardize filenames across the dataset by running:  
   ```
   python datasets/handleIEMOCAP.py
   ```

---

## Feature Extraction

The repository provides scripts to extract multiple types of features from speech signals:

- **Log Mel Spectrogram**:  
  ```
  python feature_extraction/extract_features_iemocap_mel_spectrogram.py
  ```
- **MFCC Features**:  
  ```
  python feature_extraction/extract_features_iemocap_mfcc.py
  ```
- **Spectrogram**:  
  ```
  python feature_extraction/extract_features_iemocap_spectrogram.py
  ```

Specify the dataset paths in the scripts to extract features for both the original and augmented datasets.

---

## Model Implementation

The file `models.py` includes the implementation of:  
- **LANCET**: The proposed Lightweight Attention-enhanced CNN-based Emotion Recognition Network.  
- **Baseline Methods**: State-of-the-art CNN-based approaches, such as GLAM and TC-Net, for comparative analysis.

---

## Training and Evaluation

Train and evaluate LANCET on the IEMOCAP dataset using:  
```
python train_LANCET.py
```

Key features:
- Performs training and testing for LANCET and baseline methods.
- Reports key metrics such as weighted accuracy (WA) and unweighted accuracy (UA).

