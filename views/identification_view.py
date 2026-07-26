"""
views/identification_view.py
-------------------------------
Onglet "Identification / Vérification" : enregistre une voix et
détermine soit "qui parle" (identification, 1:N), soit "est-ce bien X"
(vérification, 1:1), avec affichage du score de confiance.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QRadioButton, QButtonGroup, QProgressBar, QMessageBox
)

import config
from utils.audio_recorder import RecordingThread
from utils.plot_utils import ProbabilityBarCanvas


class IdentificationView(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.recording_thread = None
        self._build_ui()
        self._refresh_user_combo()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(16)

        # ---- Panneau gauche : contrôles ----
        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._section_title("🔍 Reconnaissance"))

        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.radio_identify = QRadioButton("Identification (Qui parle ?)")
        self.radio_verify = QRadioButton("Vérification (Est-ce bien X ?)")
        self.radio_identify.setChecked(True)
        self.mode_group.addButton(self.radio_identify)
        self.mode_group.addButton(self.radio_verify)
        mode_row.addWidget(self.radio_identify)
        mode_row.addWidget(self.radio_verify)
        left_layout.addLayout(mode_row)
        self.radio_verify.toggled.connect(self._toggle_mode)

        left_layout.addWidget(QLabel("Identité prétendue (mode vérification) :"))
        self.user_combo = QComboBox()
        self.user_combo.setEnabled(False)
        left_layout.addWidget(self.user_combo)

        left_layout.addSpacing(10)
        self.record_btn = QPushButton(f"🎙️ Enregistrer et analyser ({config.RECORD_DURATION:.0f}s)")
        self.record_btn.clicked.connect(self._record_and_analyze)
        left_layout.addWidget(self.record_btn)

        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        left_layout.addSpacing(14)
        self.result_label = QLabel("En attente d'un enregistrement...")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        left_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.confidence_label)

        left_layout.addStretch()
        root.addWidget(left_panel, 2)

        # ---- Panneau droit : histogramme de probabilités ----
        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self._section_title("📊 Scores de confiance"))
        self.canvas = ProbabilityBarCanvas()
        right_layout.addWidget(self.canvas)
        root.addWidget(right_panel, 3)

    def _section_title(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 15px; font-weight: 700; padding: 4px 0;")
        return label

    # ------------------------------------------------------------------
    def _refresh_user_combo(self):
        self.user_combo.clear()
        for user in self.controller.get_users():
            self.user_combo.addItem(user["name"])

    def _toggle_mode(self, checked):
        self.user_combo.setEnabled(checked)

    # ------------------------------------------------------------------
    def _record_and_analyze(self):
        self._refresh_user_combo()
        if not self.controller.model.is_trained:
            QMessageBox.warning(
                self, "Modèle non entraîné",
                "Veuillez d'abord entraîner le modèle dans l'onglet Entraînement."
            )
            return
        if self.radio_verify.isChecked() and self.user_combo.count() == 0:
            QMessageBox.warning(self, "Aucun locuteur", "Aucun locuteur enrôlé.")
            return

        self.record_btn.setEnabled(False)
        self.result_label.setText("🎙️ Enregistrement en cours...")
        self.progress_bar.setValue(0)

        self.recording_thread = RecordingThread(duration=config.RECORD_DURATION)
        self.recording_thread.progress.connect(self.progress_bar.setValue)
        self.recording_thread.finished_recording.connect(self._on_recorded)
        self.recording_thread.error.connect(self._on_error)
        self.recording_thread.start()

    def _on_recorded(self, audio, _filepath):
        self.record_btn.setEnabled(True)
        try:
            if self.radio_identify.isChecked():
                self._run_identification(audio)
            else:
                self._run_verification(audio)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erreur", str(exc))

    def _run_identification(self, audio):
        best_name, best_prob, all_probs, accepted = self.controller.identify(audio)
        names = list(all_probs.keys())
        probs = [all_probs[n] for n in names]
        self.canvas.update_probabilities(names, probs, predicted_name=best_name)

        if accepted:
            self.result_label.setText(f"✅ Locuteur identifié : {best_name}")
            self.result_label.setStyleSheet(
                "font-size: 16px; font-weight: 700; color: #48BB78;"
            )
        else:
            self.result_label.setText("⚠️ Locuteur inconnu / confiance insuffisante")
            self.result_label.setStyleSheet(
                "font-size: 16px; font-weight: 700; color: #F6E05E;"
            )
        self.confidence_label.setText(f"Confiance : {best_prob:.1%}")

    def _run_verification(self, audio):
        claimed_name = self.user_combo.currentText()
        accepted, score = self.controller.verify(audio, claimed_name)
        self.canvas.update_probabilities(
            [claimed_name], [score], predicted_name=claimed_name if accepted else None
        )
        if accepted:
            self.result_label.setText(f"✅ Identité confirmée : {claimed_name}")
            self.result_label.setStyleSheet(
                "font-size: 16px; font-weight: 700; color: #48BB78;"
            )
        else:
            self.result_label.setText(f"❌ Identité rejetée pour « {claimed_name} »")
            self.result_label.setStyleSheet(
                "font-size: 16px; font-weight: 700; color: #E53E3E;"
            )
        self.confidence_label.setText(f"Similarité : {score:.1%}")

    def _on_error(self, message):
        self.record_btn.setEnabled(True)
        QMessageBox.critical(self, "Erreur d'enregistrement", message)
