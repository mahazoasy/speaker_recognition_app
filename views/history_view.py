"""
views/history_view.py
------------------------
Onglet "Historique" : affiche l'historique des tentatives
d'identification / vérification enregistrées en base SQLite.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView
)


class HistoryView(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)

        header_row = QHBoxLayout()
        title = QLabel("📈 Historique des reconnaissances")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        header_row.addWidget(title)
        header_row.addStretch()
        refresh_btn = QPushButton("🔄 Actualiser")
        refresh_btn.setObjectName("SecondaryButton")
        refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Horodatage", "Mode", "Locuteur prédit / prétendu", "Confiance", "Résultat"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        root.addWidget(panel)

    def refresh(self):
        rows = self.controller.get_recognition_history()
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            mode = "Identification" if row["mode"] == "identification" else "Vérification"
            accepted = bool(row["accepted"])
            values = [
                row["timestamp"],
                mode,
                row["predicted_name"] or "-",
                f"{row['confidence']:.1%}" if row["confidence"] is not None else "-",
                "✅ Acceptée" if accepted else "❌ Rejetée",
            ]
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                if j == 4:
                    item.setForeground(Qt.green if accepted else Qt.red)
                self.table.setItem(i, j, item)
