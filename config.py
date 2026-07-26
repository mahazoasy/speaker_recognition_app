"""
config.py
---------
Configuration centralisée du projet de Reconnaissance du Locuteur.
Toutes les constantes (audio, réseau de neurones, chemins) sont ici.
"""

import os

# ------------------------------------------------------------------
# Chemins du projet
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DB_PATH = os.path.join(DATA_DIR, "speaker_recognition.db")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "speaker_model.pt")
LABELS_PATH = os.path.join(CHECKPOINT_DIR, "labels.json")
CENTROIDS_PATH = os.path.join(CHECKPOINT_DIR, "centroids.json")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Paramètres audio
# ------------------------------------------------------------------
SAMPLE_RATE = 16000          # Fréquence d'échantillonnage (Hz)
RECORD_DURATION = 3.0        # Durée d'un enregistrement (secondes)
N_MFCC = 40                  # Nombre de coefficients MFCC
MAX_FRAMES = 200             # Longueur temporelle fixe (padding/troncature)

# Nombre d'échantillons vocaux à enregistrer par locuteur à l'enrôlement
SAMPLES_PER_SPEAKER = 5

# ------------------------------------------------------------------
# Paramètres du réseau de neurones
# ------------------------------------------------------------------
EMBEDDING_DIM = 128           # Dimension du vecteur d'empreinte vocale
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3

# Seuils de décision
IDENTIFICATION_THRESHOLD = 0.55   # Confiance minimale (softmax) pour accepter une identification
VERIFICATION_THRESHOLD = 0.75     # Similarité cosinus minimale pour vérifier une identité

# ------------------------------------------------------------------
# Interface graphique
# ------------------------------------------------------------------
APP_NAME = "VoxID — Reconnaissance du Locuteur par Réseau de Neurones"
ORG_NAME = "Universite Adventiste Zurcher"
