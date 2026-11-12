# 🌌 AudioPulse 3D — Offline 3D Audio & Music Intelligence Engine

> **100% Offline Edge Audio Intelligence, AI Demixer, MIDI Transcriber & 3D WebGL Spectrogram Laboratory**  
> Powered by Three.js, Web Audio DSP, and a custom PyTorch Convolutional Neural Network exported to ONNX Web.

[![Status](https://img.shields.io/badge/Status-Complete-00f3ff.svg)](#)
[![Hardware](https://img.shields.io/badge/Trained_on-RTX_3060_GPU-76b900.svg)](#)
[![Zero-Cloud](https://img.shields.io/badge/Cloud_APIs-Zero_(100%25_Offline)-ff007f.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#)

---

## 🌟 Overview
**AudioPulse 3D** is an autonomous, client-side music intelligence engine, AI demixer, and real-time 3D audio visualizer. It allows users to drop or load any audio file (MP3, WAV, FLAC, OGG, M4A) or synthesize new music locally, and instantaneously deconstruct its acoustic anatomy:
* **Real-time 60 FPS 3D WebGL Visualization:** Three dynamic reactive modes (Orbital Particle Nebula, Flowing Spectrogram Mountain, and Cymatics Resonance Sphere).
* **Mathematical Audio DSP Engine:** Extracts exact BPM tempo, musical key & scale (via 12-dimensional Chroma Pitch Class Profiles & Krumhansl-Schmuckler harmonic correlation), Camelot Wheel mixing codes, dynamic range, RMS loudness, and peak dBFS.
* **Multi-Head Neural AI Classifier:** A custom PyTorch CNN (`AudioPulseNet`) trained on an RTX 3060 and exported to ONNX Web for real-time offline genre classification, mood/energy analysis, and 8-instrument presence detection.
* **Neural 4-Stem Demixer & Multitrack Mixer:**
  * Separates any input audio into 4 isolated audio streams: 🎤 **Vocals**, 🥁 **Drums**, 🎸 **Bass**, and 🎹 **Melody**.
  * Dedicated **Mute (M)**, **Solo (S)**, and volume faders per stem.
  * 3D visualizer dynamically morphs based on which stems are active (muting drums halts percussive kicks; soloing vocals isolates the glowing aura).
  * 1-click offline export of separated stems as standard uncompressed `.wav` files.
* **Audio-to-MIDI Neural Transcriber & Cyberpunk Piano Roll:**
  * Uses YIN pitch tracking and spectral harmonic peak autocorrelation to transcribe musical pitches into discrete notes.
  * Transcribes from the **Master Mix**, **Melody Stem**, **Bassline Stem**, or **Vocals Stem**.
  * Interactive **Cyberpunk Piano Roll** with real-time active playhead note illumination.
  * In-browser polyphonic synth preview to audition the transcribed notes directly in Web Audio.
  * 1-click **Download .MID File** (Standard MIDI Format 0) ready for instant drag-and-drop into FL Studio, Ableton Live, Logic Pro, or Cubase.
* **Generative AI Music Studio (Offline Synthesis):**
  * Synthesizes original multi-track electronic & acoustic audio compositions locally in your browser.
  * Natural language prompt parser (e.g., *"dark synthwave in 130 bpm with punchy kick in A minor"*).
  * Quick style presets: *⚡ Synthwave Overdrive*, *🔥 Cyber Dystopia*, *🌌 Lo-Fi Cosmic Chill*, *🌿 Ambient Ethereal*, and *🥁 Urban 808 Groove*.
  * Automatically loads synthesized tracks directly into the 3D visualizer, DSP telemetry, stem separator, and MIDI transcriber.
* **Dual Intelligence Export:**
  * **Export JSON:** Full raw acoustic and neural telemetry object for programmatic analysis.
  * **Export Story Report (.txt):** In-depth natural language acoustic intelligence dossier featuring harmonic mixing guidance, tempo cadence evaluation, dynamic mastering analysis, and curator recommendations.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      AUDIOPULSE 3D EDGE PIPELINE                       │
└────────────────────────────────────────────────────────────────────────┘
                                    │
               [ Upload Any Audio File OR AI Prompt Synthesis ]
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│    1. REAL-TIME DSP & DEMIXER   │   │   2. MULTI-HEAD NEURAL AI (ONNX) │
│     (Pure Mathematical DSP)     │   │       (Genre, Mood, Instruments) │
├─────────────────────────────────┤   ├──────────────────────────────────┤
│ • 60 FPS 3D Three.js Visualizer │   │ • Mel-Spectrogram 2D Feature Map │
│ • Exact BPM / Tempo Detection   │   │ • 10-Class Genre Probabilities   │
│ • Key & Scale (Chroma Vectors)  │   │ • 8-Instrument Presence Matrix   │
│ • 4-Stem Demixer (Mute / Solo)  │   │ • Mood & Valence Radar Profile   │
│ • Audio-to-MIDI Transcription   │   │ • 100% Offline (ONNX Web / WASM) │
└─────────────────────────────────┘   └──────────────────────────────────┘
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CYBERPUNK AUDIO INTELLIGENCE HUD                     │
│   • 3D Audio Visualizer (Nebula, Spectrogram Terrain, Cymatics)        │
│   • Live Telemetry HUD (BPM, Key, Dynamic Range, Sample Rate)          │
│   • 4-Stem Channel Faders (Vocals, Drums, Bass, Melody)                │
│   • Interactive Piano Roll with .MID and .WAV Downloads                │
│   • Export Full Song Intelligence Report (.json & .txt Story Dossier)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Running Locally

To run AudioPulse 3D locally on your machine:

```bash
# 1. Clone the repository
git clone https://github.com/subashis-dev/audio-pulse-3d.git
cd audio-pulse-3d

# 2. Start any standard local HTTP server
python -m http.server 8090
# or: npx serve .
```

Open your browser and navigate to:
```
http://localhost:8090
```

---

## 📊 Retraining the Neural Model (Optional)

To retrain the neural model on your own GPU:

```bash
cd train
python train.py
```
This trains the multi-head architecture on CUDA and automatically exports the updated ONNX model to `models/`.

---

## 📜 License
MIT License. Built for local sovereign edge intelligence.
