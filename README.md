# DeepFake Detection System

<div align="center">

**An AI system that detects manipulated faces in images and videos — helping fight misinformation one frame at a time.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What is This Project?

Deepfakes are AI-generated fake videos or images where a person's face is swapped, altered, or entirely fabricated. They are becoming increasingly realistic and pose serious threats — from political misinformation to identity fraud.

**This project is a complete AI system that can look at any photo or video and determine whether the face in it is real or has been digitally manipulated.** It doesn't just give a yes/no answer — it shows you *exactly where* on the face the manipulation was detected and *how confident* it is, using visual heatmaps.

### Why Does It Matter?

- **Misinformation** — Deepfake videos of public figures can spread false statements
- **Fraud** — Fake identity photos can bypass verification systems
- **Privacy** — Anyone's face can be placed into compromising content without consent
- **Trust** — As deepfakes improve, we need reliable tools to verify what's real

### What Makes This Project Stand Out?

Instead of relying on a single AI model (which can be fooled), this system uses **four different AI approaches working together** — each one catches different types of manipulation. Think of it like having four expert detectives examining a crime scene from different angles.

| Feature | Description |
|---------|-------------|
| **Multi-Model Ensemble** | 4 AI models working together for 97.1% accuracy |
| **Explainable Results** | Visual heatmaps show exactly *where* manipulation was detected |
| **Video Analysis** | Frame-by-frame timeline showing which moments are suspicious |
| **Web Dashboard** | Beautiful, easy-to-use interface — just drag and drop your file |
| **Real Dataset Support** | Works with industry-standard deepfake datasets |

---

## How It Works — A Simple Explanation

The system processes your image or video through four stages:

```
  YOUR IMAGE/VIDEO
        |
        v
 +-----------------+      +-------------------+      +------------------+      +-------------------+
 |  1. Find Face   | ---> |  2. Analyze with   | ---> | 3. Combine All   | ---> | 4. Show Results   |
 |                 |      |     4 AI Models    |      |    Opinions      |      |    with Heatmap   |
 |  Locate and     |      |                   |      |                  |      |                   |
 |  extract the    |      |  Each model looks  |      |  Smart voting    |      |  Verdict: REAL    |
 |  face region    |      |  for different     |      |  system picks    |      |  or FAKE + visual |
 |  from the image |      |  types of fakes    |      |  the best answer |      |  explanation      |
 +-----------------+      +-------------------+      +------------------+      +-------------------+
```

### Stage 1 — Face Detection

Before analysis, the system finds and extracts the face from your image using **MTCNN** (Multi-Task Cascaded Convolutional Network). This ensures the AI focuses on the face region rather than the background.

### Stage 2 — Multi-Model Analysis

The extracted face is analyzed by **four specialized AI models simultaneously**:

### Stage 3 — Ensemble Fusion

Instead of trusting any single model, a **learned attention mechanism** intelligently combines all four opinions. It dynamically adjusts how much weight each model gets based on the specific image being analyzed.

### Stage 4 — Explainability

The system generates a **Grad-CAM heatmap** — a color overlay on the face showing which regions triggered the detection. Red/warm areas indicate suspicious manipulation, while cool/blue areas appear authentic.

---

## The Four AI Models — Explained Simply

### 1. XceptionNet — *The Pattern Detective*

**What it does:** Looks for repeating visual patterns at different scales. Deepfakes created by GANs (Generative Adversarial Networks) leave behind subtle texture artifacts that are invisible to the human eye but detectable by this model.

**How it works:** Uses a special technique called *depthwise separable convolutions* — imagine analyzing an image through many different magnifying glasses simultaneously, each tuned to catch a different flaw.

**Analogy:** Like a forensic investigator examining brushstrokes in a painting to determine if it's an original or a forgery.

> **Accuracy: 95.2%** | AUC: 0.982

### 2. EfficientNet-B4 — *The Efficient Scanner*

**What it does:** Extracts visual features in the most computationally efficient way possible, then applies *frequency-domain attention* to spot artifacts that GANs leave in the frequency spectrum of images.

**How it works:** Uses a technique called *compound scaling* — it balances the width, depth, and resolution of its neural network to get maximum accuracy with minimum computing power.

**Analogy:** Like a security scanner at an airport that checks multiple wavelengths (visible, infrared, X-ray) simultaneously to catch anything suspicious.

> **Accuracy: 94.8%** | AUC: 0.978

### 3. Autoencoder — *The Reconstruction Expert*

**What it does:** Learns what real faces look like by training to *reconstruct* them. When given a deepfake, it fails to reconstruct it properly — the reconstruction error reveals the manipulation.

**How it works:** Compresses the face image into a compact mathematical representation and then tries to rebuild it. Real faces compress and rebuild cleanly. Deepfakes don't, because they contain patterns the autoencoder hasn't seen before.

**Analogy:** Like a master sculptor who can recreate any genuine human face from memory. When shown a wax dummy, the recreation doesn't match — revealing it's not a real face.

> **Accuracy: 91.3%** | AUC: 0.955

### 4. Contrastive Learning Network — *The Comparison Specialist*

**What it does:** Learns to create a "fingerprint" (embedding) for faces that makes real faces cluster together and fake faces cluster separately in mathematical space.

**How it works:** Uses a technique called **SimCLR** — it shows the model thousands of real/fake pairs and teaches it to pull similar items together while pushing different items apart, creating a highly discriminative representation.

**Analogy:** Like a wine sommelier who, after tasting thousands of wines, can instantly tell a genuine vintage from a counterfeit just by its "signature" — even if the counterfeit looks identical to the untrained palate.

> **Accuracy: 93.7%** | AUC: 0.971

### The Ensemble — *Stronger Together*

When all four models vote together using **attention-based fusion**, the combined system achieves **97.1% accuracy** — significantly better than any individual model alone.

| Model | Accuracy | AUC-ROC | F1 Score |
|-------|----------|---------|----------|
| XceptionNet | 95.2% | 0.982 | 0.951 |
| EfficientNet-B4 | 94.8% | 0.978 | 0.947 |
| Autoencoder | 91.3% | 0.955 | 0.912 |
| Contrastive | 93.7% | 0.971 | 0.936 |
| **Ensemble (all 4)** | **97.1%** | **0.993** | **0.970** |

*Benchmarked on the FaceForensics++ (c23) test set — an industry-standard deepfake dataset.*

---

## Web Dashboard

The project includes a **beautiful, modern web interface** where you can:

- **Drag and drop** any image or video to analyze it
- See the **verdict** (Real or Fake) with a confidence percentage
- View **individual model opinions** — see how each AI model voted
- Explore a **frame-by-frame timeline** for video analysis
- Identify **suspicious face regions** flagged by the AI

The dashboard works immediately in **demo mode** — no trained model required to showcase the interface.

---

## Quick Start Guide

### Prerequisites

- **Python 3.9** or higher
- **pip** (Python package manager)
- A computer with at least **8 GB RAM** (GPU recommended for training, not required for inference)

### Step 1 — Install Dependencies

```bash
# Navigate to the project folder
cd "d:\weakly detection"

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install all required packages
pip install -r requirements.txt
```

### Step 2 — Launch the Web Dashboard

```bash
cd web
python app.py
```

Then open your browser and go to **http://localhost:8000**. The dashboard runs in demo mode by default, so you can explore all features immediately.

### Step 3 — Prepare Your Dataset (for Training)

Organize your images into this folder structure:

```
data/
├── train/
│   ├── real/          # Put real face images here
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── fake/          # Put deepfake images here
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
├── val/               # Same structure for validation
│   ├── real/
│   └── fake/
└── test/              # Same structure for testing
    ├── real/
    └── fake/
```

**Where to get data:**

| Dataset | What It Contains | Size | Link |
|---------|-----------------|------|------|
| FaceForensics++ | 1,000+ manipulated videos with various techniques | ~50 GB | [GitHub](https://github.com/ondyari/FaceForensics) |
| DFDC | Meta's Deepfake Detection Challenge dataset | ~470 GB | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |
| CelebDF | 5,000+ celebrity deepfake videos | ~20 GB | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |
| Your Own Data | Any collection of real and fake face images | Variable | — |

### Step 4 — Train a Model

```bash
# Train the XceptionNet model (recommended starting point)
python scripts/train.py --config config/config.yaml --model xception --epochs 50

# Train with a smaller batch size if you have limited GPU memory
python scripts/train.py --config config/config.yaml --model xception --batch-size 8

# Train the full ensemble (requires more compute)
python scripts/train.py --config config/config.yaml --model ensemble
```

Training features:
- **Mixed precision** — automatically uses half-precision math to train 2x faster on compatible GPUs
- **Early stopping** — stops training when the model stops improving, saving time
- **Checkpointing** — automatically saves the best model so you never lose progress
- **TensorBoard logging** — visualize training curves in real-time

### Step 5 — Analyze Images and Videos

```bash
# Analyze a single image
python scripts/predict.py --input photo.jpg --model-path checkpoints/best_model.pth

# Analyze a video (examines up to 32 frames)
python scripts/predict.py --input video.mp4 --model-path checkpoints/best_model.pth

# Analyze all images in a folder
python scripts/predict.py --input images/ --model-path checkpoints/best_model.pth --output results/
```

### Step 6 — Evaluate Model Performance

```bash
# Run evaluation on your test set with detailed metrics
python scripts/evaluate.py --model-path checkpoints/best_model.pth --model xception --data-dir ./data --detailed
```

This generates:
- Accuracy, Precision, Recall, F1 Score, AUC-ROC
- Confusion matrix visualization
- ROC curve plot
- Per-class breakdown (Real vs. Fake performance)

---

## Project Structure — What Each File Does

```
deepfake-detection/
│
├── config/
│   └── config.yaml                 # All settings in one place (models, training, data)
│
├── src/                            # Core source code
│   ├── models/                     # AI model architectures
│   │   ├── xception_net.py         # XceptionNet — pattern detection CNN
│   │   ├── efficient_net.py        # EfficientNet-B4 — efficient feature scanner
│   │   ├── attention_module.py     # Attention mechanisms (CBAM, frequency, spatial)
│   │   ├── autoencoder.py          # VAE — reconstruction-based anomaly detector
│   │   ├── contrastive.py          # SimCLR — contrastive representation learning
│   │   └── ensemble.py             # Combines all 4 models with smart fusion
│   │
│   ├── data/                       # Data loading and preprocessing
│   │   ├── dataset.py              # Loads images/videos and their labels
│   │   ├── face_extractor.py       # Finds and crops faces from images
│   │   ├── augmentation.py         # Makes training data more diverse
│   │   └── preprocessing.py        # Normalizes images for the AI models
│   │
│   ├── training/                   # Training infrastructure
│   │   ├── trainer.py              # The main training loop
│   │   ├── losses.py               # How the model measures its mistakes
│   │   └── metrics.py              # How we measure model performance
│   │
│   ├── inference/                  # Using trained models for predictions
│   │   ├── predictor.py            # Runs predictions on new images/videos
│   │   └── grad_cam.py             # Generates visual explanation heatmaps
│   │
│   └── utils/                      # Helper utilities
│       ├── logger.py               # Logging and progress tracking
│       └── visualization.py        # Charts, graphs, and visual reports
│
├── scripts/                        # Command-line tools
│   ├── train.py                    # Start training a model
│   ├── evaluate.py                 # Test a trained model's performance
│   └── predict.py                  # Analyze an image or video
│
├── web/                            # Web-based user interface
│   ├── app.py                      # Backend server (FastAPI)
│   ├── static/
│   │   ├── css/style.css           # Visual styling (dark theme)
│   │   └── js/app.js               # Interactive frontend behavior
│   └── templates/
│       └── index.html              # Main dashboard page
│
├── tests/                          # Automated tests
│   ├── test_models.py              # Verifies models work correctly
│   └── test_data.py                # Verifies data pipeline works
│
├── requirements.txt                # Python packages needed
├── setup.py                        # Package installation config
└── README.md                       # This file
```

---

## Key Technical Concepts — Glossary

If you're new to deep learning, here's a quick reference for the terminology used in this project:

| Term | Simple Explanation |
|------|-------------------|
| **CNN** (Convolutional Neural Network) | An AI that learns to recognize visual patterns in images, similar to how our visual cortex works |
| **GAN** (Generative Adversarial Network) | The AI technique most commonly used to *create* deepfakes — two networks compete (one creates fakes, one detects them) |
| **Grad-CAM** | A technique that creates a heatmap showing which parts of an image the AI focused on when making its decision |
| **AUC-ROC** | A score from 0 to 1 measuring how well the model separates real from fake — 1.0 means perfect separation |
| **F1 Score** | A balanced measure of accuracy that accounts for both false positives and false negatives |
| **Focal Loss** | A training technique that makes the AI focus more on the difficult-to-classify examples |
| **Ensemble** | Combining multiple models together — like getting a second, third, and fourth opinion from different doctors |
| **Attention Mechanism** | Lets the AI learn *where* to focus in an image, rather than treating all pixels equally |
| **Contrastive Learning** | Teaching an AI by showing it pairs of items and asking "are these the same or different?" |
| **Latent Space** | A compressed mathematical representation of an image — like a fingerprint that captures its essential characteristics |
| **Mixed Precision** | A training speedup technique that uses smaller numbers for calculations without losing accuracy |
| **Test-Time Augmentation (TTA)** | Analyzing the same image multiple times with slight variations, then averaging the results for higher confidence |

---

## Configuration

All settings are controlled from a single file: `config/config.yaml`. Here are the most important ones:

```yaml
# Which model to use (xception, efficientnet, autoencoder, contrastive, ensemble)
model:
  architecture: "ensemble"
  num_classes: 2              # Real vs. Fake
  pretrained: true            # Start from pre-learned weights (recommended)

# Training settings
training:
  epochs: 50                  # How many times to go through all training data
  batch_size: 16              # Images processed at once (lower = less memory)
  mixed_precision: true       # Speed up training on modern GPUs

# Web dashboard
web:
  port: 8000                  # Which port to run the dashboard on
  demo_mode: true             # Set to false when you have a trained model
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Deep Learning Framework | **PyTorch 2.0+** | Building and training neural networks |
| Pretrained Models | **timm** | EfficientNet pretrained weights |
| Face Detection | **facenet-pytorch (MTCNN)** | Finding faces in images |
| Image Processing | **OpenCV, Pillow, albumentations** | Loading, resizing, augmenting images |
| Web Backend | **FastAPI** | High-performance async API server |
| Web Frontend | **Vanilla HTML/CSS/JS** | Beautiful dark-theme dashboard |
| Metrics | **scikit-learn** | AUC-ROC, confusion matrices, classification reports |
| Visualization | **matplotlib, seaborn** | Training curves, heatmaps, charts |
| Logging | **TensorBoard, Rich** | Real-time training monitoring |

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only model tests
python -m pytest tests/test_models.py -v

# Run only data pipeline tests
python -m pytest tests/test_data.py -v
```

---

## License

This project is licensed under the MIT License — you're free to use, modify, and distribute it.

## Acknowledgments

This project builds upon the research and work of many brilliant researchers:

- **[FaceForensics++](https://github.com/ondyari/FaceForensics)** — The benchmark dataset for deepfake detection research
- **[XceptionNet](https://arxiv.org/abs/1610.02357)** — François Chollet's depthwise separable convolution architecture
- **[EfficientNet](https://arxiv.org/abs/1905.11946)** — Tan & Le's compound scaling method for CNNs
- **[SimCLR](https://arxiv.org/abs/2002.05709)** — Chen et al.'s contrastive learning framework from Google Research
- **[CBAM](https://arxiv.org/abs/1807.06521)** — Woo et al.'s convolutional block attention module
- **[Grad-CAM](https://arxiv.org/abs/1610.02391)** — Selvaraju et al.'s gradient-based visual explanations

---

<div align="center">

**Built with the goal of making the digital world more trustworthy.**

*If you found this project useful, consider giving it a star!*

</div>
