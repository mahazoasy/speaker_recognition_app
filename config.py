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
# Recommandé : 10-15, dans des conditions variées (moment de la journée,
# distance au micro, intonation) pour une empreinte vocale plus robuste.
SAMPLES_PER_SPEAKER = 12

# ------------------------------------------------------------------
# Extracteur d'embeddings pré-entraîné (SpeechBrain ECAPA-TDNN)
# ------------------------------------------------------------------
# Modèle pré-entraîné sur VoxCeleb (des dizaines de milliers de locuteurs).
# Il est utilisé UNIQUEMENT en inférence (poids gelés) pour transformer
# chaque enregistrement en une empreinte vocale de 192 dimensions. Un petit
# classifieur est ensuite entraîné par-dessus sur vos locuteurs enrôlés
# (apprentissage par transfert). Nécessite une connexion internet lors du
# tout premier lancement (téléchargement automatique et mise en cache
# des poids, ~80 Mo).
PRETRAINED_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
PRETRAINED_EMBEDDING_DIM = 192

# ------------------------------------------------------------------
# Augmentation de données (appliquée pendant l'entraînement uniquement)
# ------------------------------------------------------------------
AUGMENTATIONS_PER_SAMPLE = 4   # copies augmentées générées par enregistrement
NOISE_SNR_RANGE_DB = (12, 30)  # rapport signal/bruit gaussien ajouté
VOLUME_GAIN_RANGE_DB = (-8, 8) # variation de volume
TIME_SHIFT_MAX_SEC = 0.15      # décalage temporel aléatoire max

# ------------------------------------------------------------------
# Paramètres de la tête de classification (entraînée sur les embeddings)
# ------------------------------------------------------------------
EMBEDDING_DIM = 192           # Dimension de sortie du modèle ECAPA-TDNN pré-entraîné
CLASSIFIER_HIDDEN_DIM = 128
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