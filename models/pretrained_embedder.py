import os
import threading

import numpy as np
import torch

import config


class PretrainedEmbedder:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._classifier = None  # chargement paresseux (lazy)

    def _ensure_loaded(self):
        if self._classifier is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "Le paquet 'speechbrain' est requis pour l'extracteur "
                "ECAPA-TDNN pré-entraîné. Installez-le avec : "
                "pip install speechbrain torchaudio"
            ) from exc

        savedir = os.path.join(config.CHECKPOINT_DIR, "pretrained_ecapa")
        os.makedirs(savedir, exist_ok=True)
        self._classifier = EncoderClassifier.from_hparams(
            source=config.PRETRAINED_SOURCE,
            savedir=savedir,
            run_opts={"device": str(self.device)},
        )
        # Les poids restent gelés : on ne les entraîne jamais.
        for p in self._classifier.mods.parameters():
            p.requires_grad_(False)

    def embed(self, audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
        """Transforme un signal audio mono 16kHz en un vecteur (192,)."""
        self._ensure_loaded()
        if sr != config.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr,
                                      target_sr=config.SAMPLE_RATE)

        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            embedding = self._classifier.encode_batch(tensor)
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def embed_batch(self, audio_list, sr: int = config.SAMPLE_RATE) -> np.ndarray:
        """Version batch (plus rapide pour l'entraînement)."""
        return np.stack([self.embed(a, sr) for a in audio_list])