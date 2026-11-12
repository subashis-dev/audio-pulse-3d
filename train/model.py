import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = pool
        if pool:
            self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        res = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        if res.shape[1] == x.shape[1]:
            x = x + res
        if self.pool:
            x = self.pool_layer(x)
        return x

class SonixAudioNet(nn.Module):
    """
    Lightweight Deep Spectrogram CNN for 100% Offline Audio Intelligence.
    Dual-head: predicts Genre probabilities and Mood / Acoustic attributes.
    Input: [B, 1, 128, 128] (Frequency bins x Time frames)
    """
    def __init__(self, num_genres=10, num_moods=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2) # [B, 32, 64, 64]
        )
        
        self.block1 = ConvBlock(32, 64)    # [B, 64, 32, 32]
        self.block2 = ConvBlock(64, 128)   # [B, 128, 16, 16]
        self.block3 = ConvBlock(128, 256)  # [B, 256, 8, 8]
        self.block4 = ConvBlock(256, 256, pool=False) # [B, 256, 8, 8]
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_shared = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # Head 1: Genre logits (Softmax)
        self.genre_head = nn.Linear(256, num_genres)
        
        # Head 2: Mood / Audio dimensions (Sigmoid: [energy, valence, danceability, acousticness])
        self.mood_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_moods),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        feat = self.fc_shared(x)
        
        genre_logits = self.genre_head(feat)
        mood_scores = self.mood_head(feat)
        return genre_logits, mood_scores

if __name__ == "__main__":
    model = SonixAudioNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SonixAudioNet Total Parameters: {total_params:,}")
    dummy_input = torch.randn(2, 1, 128, 128)
    genre_out, mood_out = model(dummy_input)
    print("Genre output shape:", genre_out.shape)
    print("Mood output shape:", mood_out.shape)
