import numpy as np
import librosa

import config


def extract_features(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Transforme un signal audio 1D en une matrice de caractéristiques
    de forme fixe (3 * N_MFCC, MAX_FRAMES) prête pour le CNN.
    """
    # Suppression des silences / normalisation d'amplitude
    audio = librosa.util.normalize(audio.astype(np.float32))
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) < sr * 0.3:
        # Sécurité : signal trop court (ex. silence complet)
        audio = np.pad(audio, (0, int(sr * 0.3) - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.N_MFCC, n_fft=512, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta2], axis=0)  # (3*N_MFCC, T)

    # Normalisation par caractéristique (moyenne/écart-type sur le temps)
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-8
    features = (features - mean) / std

    # Padding / troncature à une longueur temporelle fixe
    T = features.shape[1]
    if T < config.MAX_FRAMES:
        pad_width = config.MAX_FRAMES - T
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :config.MAX_FRAMES]

    return features.astype(np.float32)  # shape: (3*N_MFCC, MAX_FRAMES)


def features_from_file(filepath: str) -> np.ndarray:
    audio, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
    return extract_features(audio, sr)
