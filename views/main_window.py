"""
views/main_window.py
-----------------------
Fenêtre principale : en-tête stylisé + QTabWidget regroupant les
4 modules fonctionnels de l'application (Enrôlement, Entraînement,
Identification/Vérification, Historique).
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget, QFrame
)

import config
from controllers.main_controller import MainController
from views.enrollment_view import EnrollmentView
from views.training_view import TrainingView
from views.identification_view import IdentificationView
from views.history_view import HistoryView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1200, 760)

        self.controller = MainController()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header())

        self.tabs = QTabWidget()
        self.tabs.setContentsMargins(16, 16, 16, 16)

        self.enrollment_view = EnrollmentView(self.controller)
        self.training_view = TrainingView(self.controller)
        self.identification_view = IdentificationView(self.controller)
        self.history_view = HistoryView(self.controller)

        self.tabs.addTab(self._wrap(self.enrollment_view), "👥 Enrôlement")
        self.tabs.addTab(self._wrap(self.training_view), "🧠 Entraînement")
        self.tabs.addTab(self._wrap(self.identification_view), "🔍 Identification")
        self.tabs.addTab(self._wrap(self.history_view), "📈 Historique")

        # Rafraîchit les listes de locuteurs quand on change d'onglet
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs)

        self._apply_stylesheet()

    def _wrap(self, widget):
        """Ajoute une marge autour de chaque onglet."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(16, 16, 16, 16)
        v.addWidget(widget)
        return container

    def _build_header(self):
        header = QFrame()
        header.setObjectName("HeaderBar")
        header.setFixedHeight(70)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(24, 8, 24, 8)
        layout.setSpacing(2)

        title = QLabel("VoxID")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel(
            "Reconnaissance du Locuteur par Réseau de Neurones — "
            "Université Adventiste Zurcher"
        )
        subtitle.setObjectName("HeaderSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _on_tab_changed(self, index):
        self.enrollment_view._refresh_user_list()
        self.identification_view._refresh_user_combo()
        if self.tabs.tabText(index).startswith("📈"):
            self.history_view.refresh()

    def _apply_stylesheet(self):
        qss_path = os.path.join(config.BASE_DIR, "assets", "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
