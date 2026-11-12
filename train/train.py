import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import SonixAudioNet

GENRES = [
    "Ambient / Chill",
    "Classical",
    "Dance / EDM",
    "Hip-Hop",
    "Jazz / Soul",
    "Metal / Heavy",
    "Pop",
    "Rock",
    "Synthwave / Cyber",
    "Acoustic / Folk"
]

MOOD_LABELS = ["energy", "valence", "danceability", "acousticness"]

# Typical acoustic priors for each genre (energy, valence, danceability, acousticness)
GENRE_PRIORS = {
    0: [0.25, 0.40, 0.20, 0.70], # Ambient
    1: [0.35, 0.45, 0.15, 0.95], # Classical
    2: [0.92, 0.80, 0.90, 0.05], # Dance / EDM
    3: [0.75, 0.65, 0.85, 0.20], # Hip-Hop
    4: [0.45, 0.60, 0.55, 0.75], # Jazz
    5: [0.95, 0.35, 0.40, 0.05], # Metal
    6: [0.78, 0.75, 0.82, 0.30], # Pop
    7: [0.82, 0.55, 0.50, 0.25], # Rock
    8: [0.85, 0.70, 0.78, 0.10], # Synthwave
    9: [0.30, 0.50, 0.35, 0.90], # Acoustic
}

def generate_spectrogram_profile(genre_idx, time_steps=128, freq_bins=128):
    """
    Generates realistic 2D acoustic spectrogram signatures matching genre physics.
    freq_bins: 0 = low bass, 127 = high treble.
    time_steps: time frames along song slice.
    """
    spec = np.zeros((freq_bins, time_steps), dtype=np.float32)
    t = np.linspace(0, 4 * np.pi, time_steps)
    freqs = np.linspace(20, 20000, freq_bins)
    
    # Base thermal / background noise
    spec += np.random.normal(0.05, 0.02, (freq_bins, time_steps))
    
    if genre_idx == 0:  # Ambient
        # Smooth low-mid drone and gentle upper harmonics
        for f in [5, 12, 22]:
            spec[f:f+4, :] += 0.6 * np.abs(np.sin(t * 0.2 + np.random.rand()))
        spec += np.outer(np.exp(-freqs / 1500), np.ones(time_steps)) * 0.4

    elif genre_idx == 1:  # Classical
        # Rich harmonic stacks (fundamental + overtones), high dynamic contrast
        for harm in [8, 16, 24, 32, 48]:
            spec[harm:harm+2, :] += 0.5 * (np.sin(t * 0.5) ** 2)
        # Sparse vibrato
        spec += np.outer(np.exp(-freqs / 3000), (np.sin(t * 1.5) + 1.2) * 0.3)

    elif genre_idx == 2:  # Dance / EDM
        # Heavy sub-bass kicks on 4/4 beats
        kick_times = np.arange(0, time_steps, 16)
        for kt in kick_times:
            if kt < time_steps:
                spec[2:14, kt:min(kt+6, time_steps)] += 0.95
                spec[60:110, kt:min(kt+3, time_steps)] += 0.4 # click transient
        # 16th note synth arpeggios in mids
        spec[25:55, :] += 0.4 * (np.sin(t * 4) > 0)
        # Hi-hat sizzle
        spec[90:125, 4::8] += 0.5

    elif genre_idx == 3:  # Hip-Hop
        # 808 deep sub-bass glides and punchy snare on beats 2 & 4
        snare_times = np.arange(8, time_steps, 32)
        for st in snare_times:
            if st < time_steps:
                spec[35:80, st:min(st+8, time_steps)] += 0.8
        spec[1:8, :] += 0.7 * (np.sin(t * 0.8) + 1) # 808 sub bass

    elif genre_idx == 4:  # Jazz / Soul
        # Complex chord clusters (7th, 9th harmonics) and walking bass
        for f in [10, 18, 25, 34, 45, 60]:
            spec[f:f+3, :] += 0.35 * (np.sin(t * 1.2 + f) + 1)
        # Brush snare high-frequency texture
        spec[70:120, :] += np.random.uniform(0.1, 0.25, (50, time_steps))

    elif genre_idx == 5:  # Metal / Heavy
        # Wall of sound: dense high-gain distortion across 200Hz - 8kHz
        spec[15:95, :] += 0.75 + np.random.normal(0, 0.15, (80, time_steps))
        # Rapid double-bass pedal bursts
        spec[2:10, 0::4] += 0.85

    elif genre_idx == 6:  # Pop
        # Clean production: prominent vocal mid frequencies (1kHz - 4kHz), tight kick
        spec[30:75, :] += 0.55 * (np.sin(t * 2) ** 2)
        spec[4:12, 0::16] += 0.8 # Tight kick
        spec[80:115, 8::16] += 0.4 # Claps

    elif genre_idx == 7:  # Rock
        # Power chord bands (crunch guitar) + acoustic drum transients
        for chord in [14, 28, 42, 56]:
            spec[chord:chord+5, :] += 0.65
        spec[4:15, 0::16] += 0.75 # kick
        spec[40:70, 8::16] += 0.7 # snare

    elif genre_idx == 8:  # Synthwave / Cyber
        # Retro gated reverb, pulsing arpeggiated bassline, neon brass stabs
        bass_pulse = (np.sin(t * 8) > 0).astype(float)
        spec[6:20, :] += 0.8 * bass_pulse
        spec[45:85, :] += 0.5 * (np.sin(t * 2) > 0.5)
        spec[90:125, :] += 0.3 * np.outer(np.ones(35), bass_pulse)

    elif genre_idx == 9:  # Acoustic / Folk
        # Wooden guitar body resonance, string pluck transients
        spec[10:35, :] += 0.5 * (np.sin(t * 1.5) ** 2)
        # Periodic strum onsets
        strums = np.arange(0, time_steps, 24)
        for s in strums:
            if s < time_steps:
                spec[15:90, s:min(s+12, time_steps)] += np.outer(np.linspace(0.7, 0.1, 75), np.exp(-np.linspace(0, 3, min(12, time_steps-s))))

    # Clip and normalize into [0, 1]
    spec = np.clip(spec, 0.0, 1.0)
    return spec

class SyntheticAudioSpectrogramDataset(Dataset):
    def __init__(self, num_samples=3000):
        self.num_samples = num_samples
        self.samples = []
        for _ in range(num_samples):
            genre = np.random.randint(0, len(GENRES))
            spec = generate_spectrogram_profile(genre)
            # Add slight random augmentation (frequency pitch shift, time stretch noise)
            shift = np.random.randint(-2, 3)
            spec = np.roll(spec, shift, axis=0)
            noise = np.random.normal(0, 0.03, spec.shape)
            spec = np.clip(spec + noise, 0.0, 1.0)
            
            # Mood attributes with slight variance around genre priors
            prior = np.array(GENRE_PRIORS[genre], dtype=np.float32)
            mood = np.clip(prior + np.random.normal(0, 0.05, 4), 0.0, 1.0)
            self.samples.append((spec, genre, mood))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        spec, genre, mood = self.samples[idx]
        tensor_spec = torch.from_numpy(spec).unsqueeze(0).float() # [1, 128, 128]
        tensor_genre = torch.tensor(genre, dtype=torch.long)
        tensor_mood = torch.from_numpy(mood).float()
        return tensor_spec, tensor_genre, tensor_mood

def main():
    print("=" * 60)
    print("🎵 SONIX3D: Training Neural Spectrogram Audio Classifier")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\n[1/4] Generating Spectrogram Acoustic Dataset (3,000 samples across 10 genres)...")
    dataset = SyntheticAudioSpectrogramDataset(num_samples=3000)
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    print(f"Dataset Split: {train_size} Train | {val_size} Validation")

    print("\n[2/4] Initializing SonixAudioNet...")
    model = SonixAudioNet(num_genres=len(GENRES), num_moods=len(MOOD_LABELS)).to(device)
    
    criterion_genre = nn.CrossEntropyLoss()
    criterion_mood = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    epochs = 15
    print(f"\n[3/4] Training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for specs, genres, moods in train_loader:
            specs = specs.to(device)
            genres = genres.to(device)
            moods = moods.to(device)
            
            optimizer.zero_grad()
            pred_genres, pred_moods = model(specs)
            
            loss_g = criterion_genre(pred_genres, genres)
            loss_m = criterion_mood(pred_moods, moods)
            loss = loss_g + 2.0 * loss_m
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * specs.size(0)
            preds = pred_genres.argmax(dim=1)
            correct += (preds == genres).sum().item()
            total += genres.size(0)
            
        scheduler.step()
        train_acc = correct / total * 100.0
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for specs, genres, moods in val_loader:
                specs = specs.to(device)
                genres = genres.to(device)
                pred_genres, _ = model(specs)
                preds = pred_genres.argmax(dim=1)
                val_correct += (preds == genres).sum().item()
                val_total += genres.size(0)
        
        val_acc = val_correct / val_total * 100.0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # Export to ONNX
    print("\n[4/4] Exporting Model to ONNX format for 100% Offline Web Execution...")
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128, device=device)
    
    os.makedirs(r"C:\Users\subas\Claude\sonix3d\models", exist_ok=True)
    onnx_path = r"C:\Users\subas\Claude\sonix3d\models\sonix_classifier.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["genre_logits", "mood_scores"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "genre_logits": {0: "batch_size"},
            "mood_scores": {0: "batch_size"}
        }
    )
    print(f"✅ ONNX model successfully saved to: {onnx_path}")
    print(f"   Size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

    # Save classes.json
    metadata = {
        "model_name": "SonixAudioNet-v1",
        "input_shape": [1, 1, 128, 128],
        "genres": GENRES,
        "moods": MOOD_LABELS,
        "sample_rate": 44100,
        "fft_size": 2048,
        "mel_bins": 128
    }
    meta_path = r"C:\Users\subas\Claude\sonix3d\models\classes.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {meta_path}")
    print("\n🎉 Training & Export Complete!")

if __name__ == "__main__":
    main()
