import os
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

import config
from database.db_manager import DBManager
from models.speaker_model import SpeakerModel


class _StopFlag:
    """Petit objet mutable partagé pour permettre d'interrompre
    l'entraînement en cours (voir SpeakerModel.train -> stop_flag)."""
    stopped = False


class MainController:
    def __init__(self):
        self.db = DBManager()
        self.model = SpeakerModel()
        self.model.load()  # charge un modèle existant s'il y en a un

        self._stop_flag = None

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
    # Entraînement (dans le thread principal, UI tenue à jour via processEvents)
    # ------------------------------------------------------------------
    def start_training(self, epochs, batch_size, lr, on_epoch, on_finished, on_log):
        records = self.db.get_all_recordings_with_names()
        n_speakers = len({r["name"] for r in records})
        if n_speakers < 2:
            on_finished(
                False,
                "Il faut au moins 2 locuteurs enrôlés (avec enregistrements) "
                "pour lancer l'entraînement."
            )
            return

        self.db.clear_training_history()
        on_log(f"Entraînement démarré sur {len(records)} enregistrements, "
               f"{n_speakers} locuteurs.")
        QApplication.processEvents()

        self._stop_flag = _StopFlag()

        def callback(epoch, loss, acc, val_loss, val_acc):
            self.db.log_training_epoch(epoch, loss, acc, val_loss, val_acc)
            on_epoch(epoch, loss, acc, val_loss, val_acc)
            # Garde l'interface réactive (barre de progression, courbes,
            # bouton Arrêter) sans faire tourner l'entraînement dans un
            # thread séparé.
            QApplication.processEvents()

        def wrapped_log(message):
            on_log(message)
            QApplication.processEvents()

        try:
            self.model.train(
                records,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                epoch_callback=callback,
                stop_flag=self._stop_flag,
                log_callback=wrapped_log,
            )
            on_finished(True, "Entraînement terminé et modèle sauvegardé avec succès.")
        except Exception as exc:  # noqa: BLE001
            on_finished(False, f"Erreur pendant l'entraînement : {exc}")

    def stop_training(self):
        if self._stop_flag is not None:
            self._stop_flag.stopped = True

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