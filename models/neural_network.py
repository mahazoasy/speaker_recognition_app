"""
models/neural_network.py
--------------------------
Architecture du réseau de neurones : un CNN 1D convolutionnel inspiré
des architectures type TDNN (Time-Delay Neural Network, dont s'inspire
ECAPA-TDNN) appliqué sur des trames MFCC + delta + delta-delta.

Le réseau produit deux sorties :
  - un "embedding" (empreinte vocale) de dimension EMBEDDING_DIM,
    utilisé pour la VÉRIFICATION (similarité cosinus entre deux voix).
  - des logits de classification, utilisés pour l'IDENTIFICATION
    (à quel locuteur enrôlé appartient cette voix, en probabilité).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SEBlock(nn.Module):
    """Squeeze-and-Excitation : pondère les canaux selon leur importance
    (même principe que dans ECAPA-TDNN)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, T)
        z = x.mean(dim=2)                 # squeeze -> (B, C)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))    # excitation -> (B, C)
        return x * z.unsqueeze(2)


class TDNNBlock(nn.Module):
    """Bloc Conv1D + BatchNorm + ReLU (brique de base d'un TDNN)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SpeakerEmbeddingNet(nn.Module):
    """
    Réseau principal de reconnaissance du locuteur.

    Entrée : (batch, 3*N_MFCC, MAX_FRAMES)
    Sorties : embedding (batch, EMBEDDING_DIM), logits (batch, num_classes)
    """

    def __init__(self, in_channels=3 * config.N_MFCC,
                 embedding_dim=config.EMBEDDING_DIM, num_classes=2):
        super().__init__()

        self.tdnn1 = TDNNBlock(in_channels, 256, kernel_size=5, dilation=1)
        self.tdnn2 = TDNNBlock(256, 256, kernel_size=3, dilation=2)
        self.tdnn3 = TDNNBlock(256, 256, kernel_size=3, dilation=3)
        self.se = SEBlock(256)
        self.tdnn4 = TDNNBlock(256, 384, kernel_size=1, dilation=1)

        # Attentive statistics pooling simplifié (moyenne + écart-type)
        self.embedding_fc = nn.Sequential(
            nn.Linear(384 * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def _stats_pool(self, x):
        # x: (B, C, T) -> concatène moyenne et écart-type sur le temps
        mean = x.mean(dim=2)
        std = x.std(dim=2) + 1e-5
        return torch.cat([mean, std], dim=1)

    def forward(self, x):
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.se(x)
        x = self.tdnn4(x)

        pooled = self._stats_pool(x)              # (B, 768)
        embedding = self.embedding_fc(pooled)      # (B, embedding_dim)
        logits = self.classifier(embedding)         # (B, num_classes)
        return embedding, logits

    def resize_classifier(self, num_classes):
        """Recrée la tête de classification si le nombre de locuteurs change."""
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.embedding_dim, num_classes),
        )
        self.num_classes = num_classes
