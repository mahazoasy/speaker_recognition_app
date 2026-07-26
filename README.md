# VoxID — Reconnaissance du Locuteur par Réseau de Neurones

Projet universitaire : application de bureau complète permettant
d'enrôler des locuteurs, d'entraîner un réseau de neurones (CNN de
type TDNN, inspiré d'ECAPA-TDNN) sur leurs empreintes vocales, puis de
les **identifier** (1:N) ou de les **vérifier** (1:1) en temps réel.

## Fonctionnalités

- Interface graphique moderne en PySide6 (thème sombre, accent orange)
- Enregistrement audio en temps réel (microphone), sans geler l'UI (QThread)
- Réseau de neurones PyTorch : blocs TDNN dilatés + Squeeze-and-Excitation
  + pooling statistique → empreinte vocale (embedding) + classification
- Enrôlement multi-utilisateurs avec plusieurs échantillons par locuteur
- Identification (qui parle ?) avec probabilités par locuteur
- Vérification (est-ce bien X ?) par similarité cosinus à l'empreinte moyenne
- Base de données SQLite (utilisateurs, enregistrements, historiques)
- Courbes d'entraînement (loss/accuracy) affichées en direct
- Historique des reconnaissances consultable dans l'application
- Architecture **MVC** : `models/`, `views/`, `controllers/`, `database/`, `utils/`

## Structure du projet

```
speaker_recognition_app/
├── main.py                      # Point d'entrée
├── config.py                    # Constantes globales
├── requirements.txt
├── database/
│   └── db_manager.py            # Accès SQLite
├── models/
│   ├── neural_network.py        # Architecture PyTorch (TDNN + SE)
│   └── speaker_model.py         # Entraînement / inférence / persistance
├── controllers/
│   └── main_controller.py       # Logique métier + thread d'entraînement
├── views/
│   ├── main_window.py           # Fenêtre principale + onglets
│   ├── enrollment_view.py       # Onglet Enrôlement
│   ├── training_view.py         # Onglet Entraînement
│   ├── identification_view.py   # Onglet Identification/Vérification
│   └── history_view.py          # Onglet Historique
├── utils/
│   ├── audio_recorder.py        # Enregistrement micro (QThread)
│   ├── feature_extractor.py     # MFCC + delta + delta-delta
│   └── plot_utils.py            # Canvas matplotlib pour Qt
├── assets/
│   └── style.qss                # Feuille de style Qt
├── checkpoints/                 # Modèle entraîné (.pt) + labels + centroïdes
└── data/
    ├── speaker_recognition.db   # Base SQLite
    └── audio/                   # Enregistrements .wav par utilisateur
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

>`torch` est volumineux : pour une installation CPU uniquement plus
> rapide, vous pouvez d'abord installer la version CPU officielle :
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Lancement

```bash
python main.py
```

## Guide d'utilisation

1. **Onglet Enrôlement** : saisissez le nom d'un locuteur, cliquez sur
   « Démarrer l'enrôlement » puis enregistrez les échantillons demandés
   (5 par défaut, modifiable dans `config.py`). Répétez pour au moins
   2 locuteurs.
2. **Onglet Entraînement** : ajustez éventuellement les hyperparamètres
   (époques, batch size, taux d'apprentissage) puis cliquez sur
   « Lancer l'entraînement ». Les courbes de loss/accuracy s'affichent
   en direct. Le modèle est sauvegardé automatiquement à la fin.
3. **Onglet Identification** :
   - Mode *Identification* : enregistrez une voix, l'application indique
     quel locuteur enrôlé correspond le mieux, avec un histogramme des
     probabilités.
   - Mode *Vérification* : choisissez une identité prétendue, enregistrez
     une voix, l'application confirme ou rejette selon un score de
     similarité cosinus.
4. **Onglet Historique** : consultez toutes les tentatives passées.

## Détails techniques du réseau de neurones

- **Entrée** : MFCC (40 coefficients) + delta + delta-delta → matrice
  `(120, 200)` normalisée (moyenne/écart-type par canal).
- **Blocs TDNN** : convolutions 1D à dilatation croissante (1, 2, 3),
  capturant un contexte temporel de plus en plus large — c'est le
  principe central des architectures de type ECAPA-TDNN utilisées en
  recherche du locuteur.
- **Squeeze-and-Excitation** : pondère l'importance de chaque canal
  de caractéristiques.
- **Pooling statistique** : concatène moyenne et écart-type temporels
  pour obtenir un vecteur de taille fixe, quel que soit la durée audio.
- **Tête d'embedding** : projection linéaire + BatchNorm → vecteur de
  128 dimensions (l'« empreinte vocale »).
- **Tête de classification** : couche linéaire → softmax sur les
  locuteurs enrôlés (utilisée pour l'identification).
- **Vérification** : similarité cosinus entre l'embedding calculé et le
  centroïde (moyenne des embeddings) du locuteur prétendu.

## Notes

- Le micro doit être accessible au système (permissions du système
  d'exploitation) — testez avec `python -c "import sounddevice as sd; print(sd.query_devices())"`
  si l'enregistrement échoue.
- Les seuils `IDENTIFICATION_THRESHOLD` et `VERIFICATION_THRESHOLD`
  (dans `config.py`) peuvent être ajustés selon vos résultats.
- Pour un projet académique, il est recommandé d'enrôler au moins
  3 à 5 locuteurs avec 5 à 10 échantillons chacun, dans un environnement
  calme, afin d'obtenir une bonne précision d'entraînement.
