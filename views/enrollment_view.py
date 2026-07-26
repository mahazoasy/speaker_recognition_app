from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QMessageBox
)

import config
from utils.audio_recorder import RecordingThread


class EnrollmentView(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.current_user_id = None
        self.current_user_name = None
        self.sample_index = 0
        self.recording_thread = None

        self._build_ui()
        self._refresh_user_list()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(16)

        # ---- Panneau gauche : liste des locuteurs enrôlés ----
        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._section_title("👥 Locuteurs enrôlés"))

        self.user_list = QListWidget()
        left_layout.addWidget(self.user_list)

        delete_btn = QPushButton("Supprimer le locuteur sélectionné")
        delete_btn.setObjectName("DangerButton")
        delete_btn.clicked.connect(self._delete_selected_user)
        left_layout.addWidget(delete_btn)

        root.addWidget(left_panel, 2)

        # ---- Panneau droit : nouvel enrôlement ----
        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self._section_title("🎤 Nouvel enrôlement"))

        form_row = QHBoxLayout()
        form_row.addWidget(QLabel("Nom du locuteur :"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ex : Diary Rakoto")
        form_row.addWidget(self.name_input)
        right_layout.addLayout(form_row)

        self.start_btn = QPushButton("Démarrer l'enrôlement")
        self.start_btn.clicked.connect(self._start_enrollment)
        right_layout.addWidget(self.start_btn)

        right_layout.addSpacing(10)
        self.sample_label = QLabel(
            f"Échantillon 0 / {config.SAMPLES_PER_SPEAKER}"
        )
        self.sample_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.sample_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        self.record_btn = QPushButton(f"🔴 Enregistrer ({config.RECORD_DURATION:.0f}s)")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self._record_sample)
        right_layout.addWidget(self.record_btn)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        right_layout.addStretch()
        instructions = QLabel(
            "💡 Conseil : enregistrez chaque échantillon dans un environnement "
            "calme, en parlant naturellement pendant toute la durée. "
            f"{config.SAMPLES_PER_SPEAKER} échantillons par locuteur donnent de "
            "meilleurs résultats d'entraînement."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #9aa4b2; font-size: 12px;")
        right_layout.addWidget(instructions)

        root.addWidget(right_panel, 3)

    def _section_title(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 15px; font-weight: 700; padding: 4px 0;")
        return label

    # ------------------------------------------------------------------
    def _refresh_user_list(self):
        self.user_list.clear()
        for user in self.controller.get_users():
            count = self.controller.recordings_count(user["id"])
            item = QListWidgetItem(f"{user['name']}   ({count} échantillons)")
            item.setData(Qt.UserRole, user["id"])
            self.user_list.addItem(item)

    def _delete_selected_user(self):
        item = self.user_list.currentItem()
        if not item:
            return
        user_id = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Confirmation",
            "Supprimer ce locuteur et tous ses enregistrements ?",
        )
        if reply == QMessageBox.Yes:
            self.controller.delete_user(user_id)
            self._refresh_user_list()

    # ------------------------------------------------------------------
    def _start_enrollment(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Nom manquant", "Veuillez saisir un nom.")
            return
        try:
            user_id = self.controller.enroll_user(name)
        except ValueError as exc:
            QMessageBox.warning(self, "Erreur", str(exc))
            return

        self.current_user_id = user_id
        self.current_user_name = name
        self.sample_index = self.controller.recordings_count(user_id)
        self.sample_label.setText(
            f"Échantillon {self.sample_index} / {config.SAMPLES_PER_SPEAKER}"
        )
        self.record_btn.setEnabled(True)
        self.status_label.setText(f"Locuteur « {name} » prêt. Cliquez pour enregistrer.")
        self.status_label.setObjectName("StatusOK")
        self.status_label.setStyleSheet("")

    def _record_sample(self):
        if self.current_user_id is None:
            return
        self.record_btn.setEnabled(False)
        self.status_label.setText("🎙️ Enregistrement en cours... parlez maintenant.")
        self.progress_bar.setValue(0)

        self.recording_thread = RecordingThread(duration=config.RECORD_DURATION)
        self.recording_thread.progress.connect(self.progress_bar.setValue)
        self.recording_thread.finished_recording.connect(self._on_sample_recorded)
        self.recording_thread.error.connect(self._on_recording_error)
        self.recording_thread.start()

    def _on_sample_recorded(self, audio, _filepath):
        self.sample_index += 1
        self.controller.register_recording(self.current_user_id, audio, self.sample_index)
        self.sample_label.setText(
            f"Échantillon {self.sample_index} / {config.SAMPLES_PER_SPEAKER}"
        )
        self.status_label.setText(f"Échantillon {self.sample_index} enregistré avec succès.")
        self.record_btn.setEnabled(True)
        self._refresh_user_list()

        if self.sample_index >= config.SAMPLES_PER_SPEAKER:
            self.status_label.setText(
                f"🎉 Enrôlement de « {self.current_user_name} » terminé "
                f"({self.sample_index} échantillons) !"
            )
            self.record_btn.setEnabled(False)
            self.name_input.clear()
            self.current_user_id = None

    def _on_recording_error(self, message):
        self.record_btn.setEnabled(True)
        QMessageBox.critical(self, "Erreur d'enregistrement", message)
