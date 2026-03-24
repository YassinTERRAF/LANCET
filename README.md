# LANCET: Lightweight Attention-enhanced CNN-based Emotion Recognition Network

This repository provides the official implementation of **LANCET**, a lightweight deep learning architecture for **speech emotion recognition (SER)** under both clean and noisy conditions.

---

## 🔍 Overview

LANCET combines:
- **Multiscale Channel-wise Attention**
- **Temporal Convolutional Networks (TCN)**
- **Frame-wise Attention**

to effectively capture both **spectral** and **temporal** dependencies in speech signals.

The model is designed to be **efficient, robust, and scalable**, making it suitable for real-world emotion recognition tasks in challenging acoustic environments.

---

## 📂 Repository Structure

```
LANCET/
├── datasets/                      # Dataset preparation and augmentation scripts
├── feature_extraction/           # Feature extraction scripts
├── models.py                     # LANCET and baseline implementations
├── train_LANCET.py               # Training and evaluation script
```

---

## 📥 Dataset Preparation

### 1. Download IEMOCAP

The IEMOCAP dataset is available for academic use:

👉 https://sail.usc.edu/iemocap/iemocap_release.htm  

After downloading, place it in:

```
datasets/IEMOCAP/
```

---

### 2. Data Augmentation (Optional)

To simulate noisy environments:

```bash
python datasets/augment_IEMOCAP.py
```

This script:
- Adds noise (babble, music, ambient, etc.)
- Supports multiple SNR levels
- Saves outputs to `datasets/IEMOCAP_augmented/`

Then standardize filenames:

```bash
python datasets/handleIEMOCAP.py
```

---

## 🚀 Feature Extraction

Extract features using the provided scripts:

```bash
# Log-Mel Spectrogram
python feature_extraction/extract_features_iemocap_mel_spectrogram.py

# MFCC
python feature_extraction/extract_features_iemocap_mfcc.py

# Spectrogram
python feature_extraction/extract_features_iemocap_spectrogram.py
```

Make sure to configure dataset paths inside each script.

---

## 🧠 Model Implementation

The file `models.py` includes:

- **LANCET** (Proposed Model)  
- **Baseline Models** (e.g., GLAM, TC-Net)

These implementations allow direct comparison with state-of-the-art methods.

---

## 🏋️ Training and Evaluation

Train and evaluate models using:

```bash
python train_LANCET.py
```

### Evaluation Metrics
- **Weighted Accuracy (WA)**
- **Unweighted Accuracy (UA)**

The script supports:
- Training on clean and noisy datasets
- Evaluation of both LANCET and baseline models

---

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@INPROCEEDINGS{11226051,
  author={Terraf, Yassin and Iraqi, Youssef},
  booktitle={2025 33rd European Signal Processing Conference (EUSIPCO)}, 
  title={LANCET: Lightweight Attention-Enhanced Network for Robust Speech Emotion Recognition}, 
  year={2025},
  pages={371-375},
  keywords={Emotion recognition;Attention mechanisms;Computational modeling;Speech recognition;Transformers;Feature extraction;Acoustics;Recording;Convolutional neural networks;Speech processing;Speech Emotion Recognition;Attention Mechanism;Temporal Convolutional Network;Noise Robustness;Challenging Acoustic Environments},
  doi={10.23919/EUSIPCO63237.2025.11226051}
}
```

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests.

---

## 📄 License

This project is released under the MIT License.

---

## 📬 Contact

- 📧 Email: yassin.terraf@um6p.ma  
- 🔗 LinkedIn: https://www.linkedin.com/in/yassin-terraf-206597151/
