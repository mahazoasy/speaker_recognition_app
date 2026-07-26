"""
controllers/main_controller.py
---------------------------------
Contrôleur (au sens MVC) : orchestre les interactions entre les vues
PySide6, la base de données SQLite et le modèle de réseau de neurones.
Contient aussi le QThread d'entraînement pour ne pas geler l'UI.
"""

import os
from PySide6.QtCore import QObject, QThread, Signal

import config
from database.db_manager import DBManager
from models.speaker_model import SpeakerModel


class TrainingWorker(QObject):
    """Exécute l'entraînement dans un thread séparé et transmet la
    progression à l'interface via des signaux Qt."""

    epoch_done = Signal(int, float, float, object, object)  # epoch, loss, acc, val_loss, val_acc
    finished = Signal(bool, str)   # succès, message
    log = Signal(str)

    def __init__(self, model: SpeakerModel, db: DBManager, epochs, batch_size, lr):
        super().__init__()
        self.model = model
        self.db = db
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.stopped = False

    def run(self):
        try:
            records = self.db.get_all_recordings_with_names()
            n_speakers = len({r["name"] for r in records})
            if n_speakers < 2:
                self.finished.emit(
                    False,
                    "Il faut au moins 2 locuteurs enrôlés (avec enregistrements) "
                    "pour lancer l'entraînement."
                )
                return

            self.db.clear_training_history()
            self.log.emit(f"Entraînement démarré sur {len(records)} enregistrements, "
                           f"{n_speakers} locuteurs.")

            def callback(epoch, loss, acc, val_loss, val_acc):
                self.db.log_training_epoch(epoch, loss, acc, val_loss, val_acc)
                self.epoch_done.emit(epoch, loss, acc, val_loss, val_acc)

            self.model.train(
                records,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                epoch_callback=callback,
                stop_flag=self,
            )
            self.finished.emit(True, "Entraînement terminé et modèle sauvegardé avec succès.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, f"Erreur pendant l'entraînement : {exc}")


class MainController:
    def __init__(self):
        self.db = DBManager()
        self.model = SpeakerModel()
        self.model.load()  # charge un modèle existant s'il y en a un

        self._train_thread = None
        self._train_worker = None

    # ------------------------------------------------------------------
    # Utilisateurs / Enrôlement
    # ------------------------------------------------------------------
    def enroll_user(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Le nom ne peut pas être vide.")
        existing = self.db.get_user_by_name(name)
        if existing:
            return existing["id"]
        return self.db.add_user(name)

    def register_recording(self, user_id: int, audio, sample_index: int) -> str:
        """Sauvegarde un enregistrement audio et l'associe à l'utilisateur en base."""
        import soundfile as sf
        user_dir = os.path.join(config.AUDIO_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        filepath = os.path.join(user_dir, f"sample_{sample_index}.wav")
        sf.write(filepath, audio, config.SAMPLE_RATE)
        self.db.add_recording(user_id, filepath)
        return filepath

    def get_users(self):
        return self.db.get_users()

    def recordings_count(self, user_id: int) -> int:
        return len(self.db.get_recordings_by_user(user_id))

    def delete_user(self, user_id: int):
        self.db.delete_user(user_id)

    # ------------------------------------------------------------------
    # Entraînement (asynchrone)
    # ------------------------------------------------------------------
    def start_training(self, epochs, batch_size, lr, on_epoch, on_finished, on_log):
        self._train_thread = QThread()
        self._train_worker = TrainingWorker(self.model, self.db, epochs, batch_size, lr)
        self._train_worker.moveToThread(self._train_thread)

        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.epoch_done.connect(on_epoch)
        self._train_worker.log.connect(on_log)

        def _cleanup(success, message):
            on_finished(success, message)
            self._train_thread.quit()

        self._train_worker.finished.connect(_cleanup)
        self._train_thread.finished.connect(self._train_thread.deleteLater)

        self._train_thread.start()

    def stop_training(self):
        if self._train_worker:
            self._train_worker.stopped = True

    def get_training_history(self):
        return self.db.get_training_history()

    # ------------------------------------------------------------------
    # Identification / Vérification
    # ------------------------------------------------------------------
    def identify(self, audio):
        if not self.model.is_trained:
            raise RuntimeError("Aucun modèle entraîné. Rendez-vous dans l'onglet Entraînement.")
        best_name, best_prob, all_probs, _ = self.model.predict(audio)
        accepted = best_prob >= config.IDENTIFICATION_THRESHOLD
        self.db.log_recognition("identification", best_name, best_prob, accepted)
        return best_name, best_prob, all_probs, accepted

    def verify(self, audio, claimed_name):
        if not self.model.is_trained:
            raise RuntimeError("Aucun modèle entraîné. Rendez-vous dans l'onglet Entraînement.")
        accepted, score = self.model.verify(audio, claimed_name)
        self.db.log_recognition("verification", claimed_name, score, accepted)
        return accepted, score

    def get_recognition_history(self):
        return self.db.get_recognition_history()
