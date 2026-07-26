import numpy as np

import config


def add_gaussian_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Ajoute un bruit blanc gaussien avec un rapport signal/bruit donné (dB)."""
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=audio.shape)
    return (audio + noise).astype(np.float32)


def change_volume(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Applique un gain (positif ou négatif) en décibels."""
    gain = 10 ** (gain_db / 20)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def time_shift(audio: np.ndarray, shift_max_sec: float, sr: int) -> np.ndarray:
    """Décale circulairement le signal dans le temps."""
    max_shift = max(1, int(shift_max_sec * sr))
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift).astype(np.float32)


def augment_audio(audio: np.ndarray, sr: int = config.SAMPLE_RATE,
                   n_augments: int = config.AUGMENTATIONS_PER_SAMPLE):
    """
    Retourne une liste contenant l'audio original suivi de `n_augments`
    variantes augmentées (combinaisons aléatoires de bruit / volume /
    décalage temporel).
    """
    variants = [audio.astype(np.float32)]
    for _ in range(n_augments):
        a = audio.astype(np.float32).copy()
        if np.random.rand() < 0.75:
            snr = np.random.uniform(*config.NOISE_SNR_RANGE_DB)
            a = add_gaussian_noise(a, snr)
        if np.random.rand() < 0.75:
            gain = np.random.uniform(*config.VOLUME_GAIN_RANGE_DB)
            a = change_volume(a, gain)
        if np.random.rand() < 0.5:
            a = time_shift(a, config.TIME_SHIFT_MAX_SEC, sr)
        variants.append(a)
    return variants