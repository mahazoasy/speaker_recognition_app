import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QThread, Signal

import config


class RecordingThread(QThread):
    """Thread Qt qui enregistre l'audio du microphone sans geler l'UI."""

    progress = Signal(int)          # pourcentage 0-100
    finished_recording = Signal(object, str)   # (np.ndarray, filepath ou None)
    error = Signal(str)

    def __init__(self, duration=config.RECORD_DURATION,
                 samplerate=config.SAMPLE_RATE, save_path=None, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.samplerate = samplerate
        self.save_path = save_path
        self._audio = None

    def run(self):
        try:
            n_samples = int(self.duration * self.samplerate)
            recording = np.zeros((n_samples, 1), dtype="float32")
            chunk = max(1, self.samplerate // 20)  # mise à jour ~20x/sec

            def callback(indata, frames, time_info, status):
                pass

            stream = sd.InputStream(
                samplerate=self.samplerate, channels=1, dtype="float32"
            )
            stream.start()
            written = 0
            while written < n_samples:
                to_read = min(chunk, n_samples - written)
                data, overflow = stream.read(to_read)
                recording[written:written + to_read] = data
                written += to_read
                self.progress.emit(int(100 * written / n_samples))
            stream.stop()
            stream.close()

            audio = recording.flatten()
            filepath = None
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                sf.write(self.save_path, audio, self.samplerate)
                filepath = self.save_path

            self.finished_recording.emit(audio, filepath)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


def load_wav(filepath, target_sr=config.SAMPLE_RATE):
    """Charge un fichier wav et le rééchantillonne si nécessaire."""
    import librosa
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return audio
