"""
utils/plot_utils.py
---------------------
Canvas matplotlib réutilisables, intégrés dans les widgets PySide6,
pour les courbes d'entraînement (loss/accuracy) et l'histogramme
des probabilités de reconnaissance.
"""

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ACCENT = "#FF6A3D"
ACCENT_DARK = "#1a2332"
GRID_COLOR = "#3a3f4b"
FG_COLOR = "#e8e8e8"


class TrainingCurveCanvas(FigureCanvas):
    """Affiche deux courbes empilées : Loss et Accuracy au fil des époques."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=ACCENT_DARK)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_loss = self.fig.add_subplot(211)
        self.ax_acc = self.fig.add_subplot(212)
        self._style_axes()
        self.fig.tight_layout(pad=2.0)

    def _style_axes(self):
        for ax, title in ((self.ax_loss, "Perte (Loss)"), (self.ax_acc, "Précision (Accuracy)")):
            ax.set_facecolor(ACCENT_DARK)
            ax.set_title(title, color=FG_COLOR, fontsize=10, fontweight="bold")
            ax.tick_params(colors=FG_COLOR, labelsize=8)
            ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)

    def update_curves(self, epochs, losses, accuracies, val_losses=None, val_accuracies=None):
        self.ax_loss.clear()
        self.ax_acc.clear()
        self._style_axes()

        self.ax_loss.plot(epochs, losses, color=ACCENT, linewidth=2, label="train")
        self.ax_acc.plot(epochs, accuracies, color="#4FD1C5", linewidth=2, label="train")

        if val_losses:
            self.ax_loss.plot(epochs, val_losses, color="#F6E05E", linewidth=1.5,
                               linestyle="--", label="val")
        if val_accuracies:
            self.ax_acc.plot(epochs, val_accuracies, color="#F6E05E", linewidth=1.5,
                              linestyle="--", label="val")

        for ax in (self.ax_loss, self.ax_acc):
            ax.legend(facecolor=ACCENT_DARK, labelcolor=FG_COLOR, fontsize=7, loc="best")

        self.fig.tight_layout(pad=2.0)
        self.draw()


class ProbabilityBarCanvas(FigureCanvas):
    """Histogramme horizontal des probabilités par locuteur."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3), dpi=100, facecolor=ACCENT_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.0)

    def update_probabilities(self, names, probs, predicted_name=None):
        self.ax.clear()
        self.ax.set_facecolor(ACCENT_DARK)
        self.ax.tick_params(colors=FG_COLOR, labelsize=8)
        self.ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.6)
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        colors = [ACCENT if n == predicted_name else "#5A6273" for n in names]
        y_pos = range(len(names))
        self.ax.barh(y_pos, probs, color=colors)
        self.ax.set_yticks(list(y_pos))
        self.ax.set_yticklabels(names, color=FG_COLOR, fontsize=9)
        self.ax.set_xlim(0, 1)
        self.ax.set_xlabel("Probabilité", color=FG_COLOR, fontsize=9)
        self.ax.invert_yaxis()

        self.fig.tight_layout(pad=1.5)
        self.draw()
