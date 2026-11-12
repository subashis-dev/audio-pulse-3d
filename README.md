# 🌌 SONIX3D — Offline 3D Audio & Music Intelligence Engine

> **100% Offline Edge Audio Intelligence & Interactive 3D WebGL Spectrogram Laboratory**  
> Powered by Three.js, Web Audio DSP, and a custom PyTorch Convolutional Neural Network exported to ONNX Web.

[![Status](https://img.shields.io/badge/Status-Complete-00f3ff.svg)](#)
[![Hardware](https://img.shields.io/badge/Trained_on-RTX_3060_GPU-76b900.svg)](#)
[![Zero-Cloud](https://img.shields.io/badge/Cloud_APIs-Zero_(100%25_Offline)-ff007f.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#)

---

## 🌟 Overview
**Sonix3D** is an autonomous, client-side music intelligence engine and real-time 3D audio visualizer. It allows users to drop or load any audio file (MP3, WAV, FLAC, OGG, M4A) and instantaneously deconstruct its acoustic anatomy:
* **Real-time 60 FPS 3D WebGL Visualization:** Three dynamic reactive modes (Orbital Particle Nebula, Flowing Spectrogram Mountain, and Cymatics Resonance Sphere).
* **Mathematical Audio DSP Engine:** Extracts exact BPM tempo, musical key & scale (via 12-dimensional Chroma Pitch Class Profiles & Krumhansl-Schmuckler harmonic correlation), Camelot Wheel mixing codes, dynamic range, RMS loudness, and peak dBFS.
* **Deep Neural Spectrogram AI:** A custom 650K-parameter PyTorch CNN (`SonixAudioNet`) trained on an RTX 3060 and exported to ONNX Web for real-time offline genre classification and mood/energy analysis.
* **Zero Cloud Dependency:** Runs 100% locally in-browser via Web Audio API, WebGL, and WebAssembly. No API keys, no server costs, no telemetry tracking.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SONIX3D EDGE PIPELINE                           │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                         [ Upload Any Audio File ]
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│    1. REAL-TIME DSP ENGINE      │   │    2. NEURAL SPECTROGRAM AI      │
│     (Pure Mathematical DSP)     │   │       (ONNX / PyTorch Model)     │
├─────────────────────────────────┤   ├──────────────────────────────────┤
│ • 60 FPS 3D Three.js Visualizer │   │ • Mel-Spectrogram 2D Feature Map │
│ • Exact BPM / Tempo Detection   │   │ • Deep CNN Classifier (~650K)    │
│ • Key & Scale (Chroma Vectors)  │   │ • Music Genre Classification     │
│ • 7-Band Frequency EQ Spectrum  │   │   (Rock, EDM, Jazz, Pop, etc.)   │
│ • Loudness / RMS & Peak dB      │   │ • Mood & Valence Estimation      │
│ • Waveform scrub & cue points   │   │ • 100% Offline (ONNX Web / WASM) │
└─────────────────────────────────┘   └──────────────────────────────────┘
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CYBERPUNK AUDIO INTELLIGENCE HUD                     │
│   • 3D Audio Visualizer (Nebula, Spectrogram Terrain, Cymatics)        │
│   • Live Telemetry HUD (BPM, Key, Dynamic Range, Sample Rate)          │
│   • AI Classification Card (Genre probabilities + Mood Radar)          │
│   • Export Full Song Intelligence Report (.json)                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Modules

### 1. 🎛️ Digital Signal Processing (`js/audioDsp.js`)
* **BPM / Tempo Estimation:** Computes frame energy envelope, generates onset novelty curves via positive first derivatives, and autocorrelates across 60–180 BPM lags.
* **Key & Scale Detection:** Accumulates energy into a 12-semitone pitch class profile (Chroma vector) and calculates Pearson correlation coefficients against Krumhansl-Kessler Major and Minor tonality templates. Maps the result to standard notation and DJ Camelot Wheel codes (e.g., `8B` for C Major, `8A` for A Minor).
* **Acoustic Dynamics:** Computes Peak dBFS, RMS loudness, and Crest Factor (Dynamic Range).
* **7-Band Frequency Breakdown:** Real-time bin energy integration across Sub-Bass (20–60Hz), Bass (60–250Hz), Low-Mids (250–500Hz), Midrange (500–2kHz), Upper-Mids (2–4kHz), Presence (4–6kHz), and Brilliance (6–20kHz).

### 2. 🌌 3D WebGL Visualizer (`js/visualizer3d.js`)
* **Mode 1 — Orbital Particle Nebula:** 8,000 instanced particles in a golden spiral galaxy configuration that pulse radially on sub-bass kicks and drift through 3D harmonic fields.
* **Mode 2 — 3D Spectrogram Terrain:** A 50x50 dynamic wireframe mesh displacing rows forward each frame based on live frequency bins, creating a glowing retro synthwave landscape.
* **Mode 3 — Cymatics Geometric Sphere:** A morphing icosahedron sphere surrounded by concentric orbital rings spinning at speeds modulated by high-frequency transients.
* **Color Themes:** Cyber Neon, Emerald Matrix, Deep Violet, and Solar Gold.

### 3. 🧠 Neural Spectrogram AI (`train/` & `js/neuralClassifier.js`)
* **Architecture:** `SonixAudioNet` — 4-stage residual convolutional network with batch normalization, leaky ReLUs, global average pooling, and dual prediction heads.
* **Trained On:** NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.4).
* **Exported To:** ONNX Runtime Web (`models/sonix_classifier.onnx`, 9.3 MB) for browser execution.

---

## 🚀 Running Locally

To run Sonix3D on your local machine:

```powershell
# Navigate to the project directory
cd C:\Users\subas\Claude\sonix3d

# Start any standard local HTTP server
python -m http.server 8090
```

Open your browser and navigate to:
```
http://localhost:8090
```

---

## 📊 Retraining the Neural Model

To retrain the neural model on your own GPU:

```powershell
cd C:\Users\subas\Claude\sonix3d\train
python train.py
```
This will train `SonixAudioNet` on CUDA and automatically export the updated `sonix_classifier.onnx` into `models/`.

---

## 📜 License
MIT License. Built for local offline edge intelligence.
