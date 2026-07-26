from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QDoubleSpinBox, QFrame, QTextEdit, QMessageBox
)

import config
from utils.plot_utils import TrainingCurveCanvas


class TrainingView(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.epochs_data, self.loss_data, self.acc_data = [], [], []
        self.val_loss_data, self.val_acc_data = [], []

        self._build_ui()
        self._load_existing_history()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(16)

        # ---- Panneau gauche : paramètres ----
        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._section_title("⚙️ Paramètres d'entraînement"))

        left_layout.addWidget(QLabel("Nombre d'époques :"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(config.DEFAULT_EPOCHS)
        left_layout.addWidget(self.epochs_spin)

        left_layout.addWidget(QLabel("Taille de batch :"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(config.DEFAULT_BATCH_SIZE)
        left_layout.addWidget(self.batch_spin)

        left_layout.addWidget(QLabel("Taux d'apprentissage :"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(config.DEFAULT_LR)
        left_layout.addWidget(self.lr_spin)

        left_layout.addSpacing(10)
        self.train_btn = QPushButton("🚀 Lancer l'entraînement")
        self.train_btn.clicked.connect(self._start_training)
        left_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹ Arrêter")
        self.stop_btn.setObjectName("SecondaryButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        left_layout.addWidget(self.stop_btn)

        left_layout.addSpacing(10)
        left_layout.addWidget(self._section_title("📝 Journal"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box)

        root.addWidget(left_panel, 2)

        # ---- Panneau droit : courbes ----
        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self._section_title("📈 Courbes d'entraînement"))

        self.canvas = TrainingCurveCanvas()
        right_layout.addWidget(self.canvas)

        self.status_label = QLabel("En attente de lancement...")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        root.addWidget(right_panel, 3)

    def _section_title(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 15px; font-weight: 700; padding: 4px 0;")
        return label

    # ------------------------------------------------------------------
    def _load_existing_history(self):
        history = self.controller.get_training_history()
        if not history:
            return
        for row in history:
            self.epochs_data.append(row["epoch"])
            self.loss_data.append(row["loss"])
            self.acc_data.append(row["accuracy"])
            self.val_loss_data.append(row["val_loss"])
            self.val_acc_data.append(row["val_accuracy"])
        self.canvas.update_curves(
            self.epochs_data, self.loss_data, self.acc_data,
            self.val_loss_data, self.val_acc_data
        )
        self.status_label.setText(
            f"Dernier entraînement : {len(history)} époques enregistrées."
        )

    # ------------------------------------------------------------------
    def _start_training(self):
        n_users = len(self.controller.get_users())
        if n_users < 2:
            QMessageBox.warning(
                self, "Locuteurs insuffisants",
                "Il faut au moins 2 locuteurs enrôlés (onglet Enrôlement) "
                "avant de lancer l'entraînement."
            )
            return

        self.epochs_data, self.loss_data, self.acc_data = [], [], []
        self.val_loss_data, self.val_acc_data = [], []
        self.log_box.clear()
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Entraînement en cours...")

        self.controller.start_training(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            on_epoch=self._on_epoch,
            on_finished=self._on_finished,
            on_log=self._on_log,
        )

    def _stop_training(self):
        self.controller.stop_training()
        self._on_log("Arrêt demandé... fin après l'époque en cours.")

    def _on_log(self, message):
        self.log_box.append(message)

    def _on_epoch(self, epoch, loss, acc, val_loss, val_acc):
        self.epochs_data.append(epoch)
        self.loss_data.append(loss)
        self.acc_data.append(acc)
        self.val_loss_data.append(val_loss)
        self.val_acc_data.append(val_acc)

        self.canvas.update_curves(
            self.epochs_data, self.loss_data, self.acc_data,
            self.val_loss_data, self.val_acc_data
        )
        val_txt = f" | val_loss={val_loss:.4f} val_acc={val_acc:.2%}" if val_loss is not None else ""
        self.log_box.append(
            f"Époque {epoch}: loss={loss:.4f} acc={acc:.2%}{val_txt}"
        )
        self.status_label.setText(f"Époque {epoch} / {self.epochs_spin.value()}")

    def _on_finished(self, success, message):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(message)
        self.log_box.append(("✅ " if success else "❌ ") + message)
        if not success:
            QMessageBox.warning(self, "Entraînement", message)
