# Multimodal Emotion Recognition on CREMA-D

A multimodal speech emotion recognition framework that combines **audio**, **facial appearance**, and **facial behavior** to classify emotions from the **CREMA-D** dataset. The repository includes individual unimodal models, a traditional machine learning baseline, and a late-fusion ensemble for multimodal evaluation.

---

# Features

* 🎤 Audio emotion recognition using **Wav2Vec2**
* 😊 Facial appearance recognition using **Vision Transformer (ViT)**
* 📊 Facial behavior recognition using **OpenFace** features
* 🤖 OpenFace **MLP** and **SVM** baselines
* 🔀 Weighted late-fusion ensemble
* 👥 Speaker-independent train/test split
* ⚖️ Class-balanced training
* 📈 Classification reports and confusion matrices
* 🧪 Automated feature-set experiments for OpenFace models

---

# Repository Structure

```text
.
├── audio_wav2vec2.py        # Audio emotion classifier
├── vision_vit.py            # Vision Transformer classifier
├── csv_mlp.py               # OpenFace MLP classifier
├── csv_svm.py               # OpenFace SVM baseline
├── multi_model_eval.py      # Multimodal ensemble evaluation
│
├── raw2audio.sh             # Convert raw videos to WAV
├── csv_mlp.sh               # Run OpenFace feature ablation experiments
│
├── raw/                     # Original CREMA-D videos (.flv)
├── audio/                   # Extracted WAV files
├── openface/                # OpenFace outputs
│   ├── *.csv
│   ├── *.hog
│   └── *_aligned/
│
├── audio_wav2vec2.pt
├── csv_mlp.pt
├── vision_vit.pt
└── README.md
```

---

# Dataset

The project uses the **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**.

Each sample contains:

* Speech recording
* Video recording
* Facial images
* OpenFace facial features

Emotion labels:

| Label | Emotion |
| ----- | ------- |
| ANG   | Angry   |
| DIS   | Disgust |
| FEA   | Fear    |
| HAP   | Happy   |
| NEU   | Neutral |
| SAD   | Sad     |

All experiments use a **speaker-independent split**, where actors are divided into training and testing sets to prevent speaker leakage.

---

# Data Preparation

## 1. Convert Videos to Audio

Raw CREMA-D videos are converted into **16 kHz mono WAV** files using FFmpeg.

```bash
bash raw2audio.sh
```

The script extracts audio from every `.flv` file and saves it into the `audio/` directory.

---

## 2. Extract OpenFace Features

Run OpenFace on each video to generate:

* facial landmark CSV files
* Action Units
* head pose
* eye gaze
* aligned face images

Expected directory:

```text
openface/
    1001_DFA_HAP_XX.csv
    1001_DFA_HAP_XX.hog
    1001_DFA_HAP_XX_aligned/
        frame_det_000001.bmp
        ...
```

---

# Models

## Audio Model (`audio_wav2vec2.py`)

Uses a pretrained **facebook/wav2vec2-base** encoder.

Pipeline:

```text
Audio
   ↓
16 kHz Mono
   ↓
Wav2Vec2
   ↓
Mean Pooling
   ↓
MLP Classifier
   ↓
Emotion
```

Features:

* Frozen Wav2Vec2 encoder
* Trainable classifier head
* AdamW optimizer
* Learning-rate scheduler
* Class-weighted cross entropy

---

## Vision Model (`vision_vit.py`)

Uses OpenFace aligned face images.

Each video is converted into an image grid before being processed by a frozen Vision Transformer.

Pipeline:

```text
Aligned Frames
      ↓
Frame Sampling
      ↓
Image Grid
      ↓
ViT Encoder
      ↓
MLP Head
      ↓
Emotion
```

---

## OpenFace MLP (`csv_mlp.py`)

Uses OpenFace features extracted from CSV files.

Supported feature groups:

* AU intensity (`AU*_r`)
* AU presence (`AU*_c`)
* Head pose
* Eye gaze

Additional preprocessing:

* confidence filtering
* temporal interpolation
* optional FFT features
* feature standardization

Example:

```bash
python csv_mlp.py \
    --feature-set aur+pose+gaze \
    --n-frames 30 \
    --conf-th 0.8
```

---

## Automated Feature Ablation

The repository includes `csv_mlp.sh`, which automatically evaluates every non-empty combination of:

* AU intensity
* AU presence
* Head pose
* Eye gaze

for the selected temporal parameters.

Run:

```bash
bash csv_mlp.sh
```

This script is useful for reproducing feature-ablation experiments.

---

## OpenFace SVM

A classical machine learning baseline consisting of:

* StandardScaler
* RBF Support Vector Machine

This provides a comparison against the neural-network models.

---

## Multimodal Ensemble

`multi_model_eval.py` loads the trained audio, vision, and OpenFace checkpoints and combines their prediction probabilities using weighted late fusion.

Default weights:

```text
0.25 × Audio
0.50 × OpenFace
0.25 × Vision
```

The script evaluates:

* Audio-only
* Vision-only
* OpenFace-only
* Multimodal ensemble

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository>
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn
pip install scipy
pip install pandas
pip install pillow
pip install numpy
```

OpenFace and FFmpeg should also be installed separately.

---

# Usage

Convert videos:

```bash
bash raw2audio.sh
```

Train the audio model:

```bash
python audio_wav2vec2.py
```

Train the vision model:

```bash
python vision_vit.py
```

Train the OpenFace MLP:

```bash
python csv_mlp.py
```

Run OpenFace feature ablation:

```bash
bash csv_mlp.sh
```

Train the SVM baseline:

```bash
python csv_svm.py
```

Evaluate the multimodal ensemble:

```bash
python multi_model_eval.py
```

---

# Evaluation

Each model reports:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

The ensemble compares unimodal performance against multimodal late fusion.

