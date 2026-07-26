"""
models/speaker_model.py
--------------------------
Encapsule le cycle de vie complet du modèle de reconnaissance du
locuteur, basé sur l'APPRENTISSAGE PAR TRANSFERT :

  1. Chaque enregistrement (+ ses variantes augmentées : bruit léger,
     variation de volume, décalage temporel) est transformé en une
     empreinte vocale de 192 dimensions par le modèle ECAPA-TDNN
     pré-entraîné sur VoxCeleb (poids gelés, jamais entraînés ici).
  2. Un petit classifieur (EmbeddingClassifierHead) est entraîné
     par-dessus ces embeddings pour reconnaître VOS locuteurs enrôlés.

Ce découplage donne une bien meilleure précision qu'un entraînement
"from scratch" sur quelques enregistrements, car les embeddings ECAPA
encodent déjà une information vocale extrêmement riche apprise sur des
dizaines de milliers de locuteurs.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import config
from models.neural_network import EmbeddingClassifierHead
from models.pretrained_embedder import PretrainedEmbedder
from utils.audio_augment import augment_audio


class SpeakerModel:
    """
    API haut niveau utilisée par le contrôleur :
      - train(records, ...)
      - predict(audio) -> identification
      - verify(audio, claimed_name) -> vérification
      - save() / load()
    """

    def __init__(self):
        torch.set_num_threads(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = PretrainedEmbedder()
        self.net = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.centroids = {}  # name -> embedding moyen (liste de floats), sur audio NON augmenté
        self._feat_mean = None
        self._feat_std = None

    # ------------------------------------------------------------------
    # Préparation
    # ------------------------------------------------------------------
    def _build_label_maps(self, records):
        names = sorted({r["name"] for r in records})
        self.label_to_idx = {name: i for i, name in enumerate(names)}
        self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}

    def _ensure_net(self, num_classes):
        if self.net is None:
            self.net = EmbeddingClassifierHead(num_classes=num_classes).to(self.device)
        elif self.net.num_classes != num_classes:
            self.net.resize_classifier(num_classes)
            self.net.to(self.device)

    def _build_embedding_dataset(self, records, log_fn=None):
        """
        Charge chaque enregistrement, génère ses variantes augmentées,
        calcule l'embedding ECAPA de chacune. Retourne :
          X (N, 192) embeddings, y (N,) labels, centroids (par locuteur,
          calculés uniquement sur l'audio ORIGINAL non augmenté).
        """
        import librosa

        X, y = [], []
        centroid_sums = {name: None for name in self.label_to_idx}
        centroid_counts = {name: 0 for name in self.label_to_idx}

        total = len(records)
        for i, rec in enumerate(records):
            audio, _ = librosa.load(rec["filepath"], sr=config.SAMPLE_RATE, mono=True)
            label_idx = self.label_to_idx[rec["name"]]

            variants = augment_audio(audio, sr=config.SAMPLE_RATE)
            for j, variant in enumerate(variants):
                emb = self.embedder.embed(variant, sr=config.SAMPLE_RATE)
                X.append(emb)
                y.append(label_idx)
                if j == 0:  # variante 0 = audio original (non augmenté)
                    name = rec["name"]
                    if centroid_sums[name] is None:
                        centroid_sums[name] = emb.copy()
                    else:
                        centroid_sums[name] += emb
                    centroid_counts[name] += 1

            if log_fn:
                log_fn(f"Extraction des empreintes vocales... "
                       f"{i + 1}/{total} enregistrements traités "
                       f"({len(variants)} variantes chacun).")

        centroids = {
            name: (centroid_sums[name] / centroid_counts[name]).tolist()
            for name in centroid_sums if centroid_sums[name] is not None
        }
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), centroids

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    def train(self, records, epochs=config.DEFAULT_EPOCHS,
              batch_size=config.DEFAULT_BATCH_SIZE, lr=config.DEFAULT_LR,
              epoch_callback=None, stop_flag=None, log_callback=None):
        """
        records : liste de {filepath, name} provenant de la base de données.
        epoch_callback(epoch, loss, acc, val_loss, val_acc) : appelé à chaque époque
        log_callback(message) : messages de progression (extraction des embeddings)
        stop_flag : objet avec attribut .stopped (bool) pour interrompre proprement
        """
        if len(records) < 2:
            raise ValueError("Il faut au moins 2 enregistrements pour entraîner le modèle.")

        self._build_label_maps(records)
        num_classes = len(self.label_to_idx)
        if num_classes < 2:
            raise ValueError("Il faut au moins 2 locuteurs différents pour entraîner le modèle.")

        if log_callback:
            log_callback("Chargement du modèle ECAPA-TDNN pré-entraîné "
                          "(SpeechBrain / VoxCeleb)...")

        X, y, self.centroids = self._build_embedding_dataset(records, log_fn=log_callback)

        self._ensure_net(num_classes)

        # Normalisation (moyenne/écart-type) des embeddings pour stabiliser l'entraînement
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._feat_mean) / self._feat_std

        full_dataset = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y))

        n_val = max(1, int(0.2 * len(full_dataset))) if len(full_dataset) >= 10 else 0
        n_train = len(full_dataset) - n_val
        if n_val > 0:
            train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
        else:
            train_ds, val_ds = full_dataset, None

        train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)),
                                   shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)),
                                 shuffle=False) if val_ds else None

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        if log_callback:
            log_callback(f"Entraînement du classifieur sur {len(full_dataset)} "
                         f"empreintes vocales (dont augmentées)...")

        history = []
        for epoch in range(1, epochs + 1):
            if stop_flag is not None and getattr(stop_flag, "stopped", False):
                break

            self.net.train()
            total_loss, correct, total = 0.0, 0, 0
            for feats, labels in train_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.net(feats)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * feats.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += feats.size(0)

            train_loss = total_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            val_loss, val_acc = None, None
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                scheduler.step(val_loss)

            history.append((epoch, train_loss, train_acc, val_loss, val_acc))
            if epoch_callback:
                epoch_callback(epoch, train_loss, train_acc, val_loss, val_acc)

        self.save()
        return history

    def _evaluate(self, loader, criterion):
        self.net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for feats, labels in loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                logits = self.net(feats)
                loss = criterion(logits, labels)
                total_loss += loss.item() * feats.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += feats.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def _embed_normalized(self, audio: np.ndarray):
        emb = self.embedder.embed(audio, sr=config.SAMPLE_RATE)
        emb_norm = (emb - self._feat_mean) / self._feat_std
        return emb, emb_norm

    def predict(self, audio: np.ndarray):
        """Identification en 'closed set' : renvoie {nom: probabilité}."""
        if self.net is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné.")
        emb, emb_norm = self._embed_normalized(audio)
        tensor = torch.from_numpy(emb_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(tensor)[0].cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        result = {self.idx_to_label[i]: float(p) for i, p in enumerate(probs)}
        best_name = max(result, key=result.get)
        return best_name, result[best_name], result, emb

    def verify(self, audio: np.ndarray, claimed_name: str):
        """Vérification 1:1 via similarité cosinus à l'empreinte moyenne (ECAPA)."""
        if claimed_name not in self.centroids:
            raise ValueError(f"Aucune empreinte enregistrée pour '{claimed_name}'.")
        emb = self.embedder.embed(audio, sr=config.SAMPLE_RATE)
        centroid = np.array(self.centroids[claimed_name])
        sim = float(
            np.dot(emb, centroid) /
            (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8)
        )
        score = (sim + 1) / 2
        accepted = score >= config.VERIFICATION_THRESHOLD
        return accepted, score

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    def save(self):
        if self.net is None:
            return
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "num_classes": self.net.num_classes,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
        }, config.MODEL_PATH)
        with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.idx_to_label, f, ensure_ascii=False, indent=2)
        with open(config.CENTROIDS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.centroids, f, ensure_ascii=False, indent=2)

    def load(self):
        if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.LABELS_PATH)):
            return False
        checkpoint = torch.load(config.MODEL_PATH, map_location=self.device, weights_only=False)
        self.idx_to_label = {
            int(k): v for k, v in json.load(open(config.LABELS_PATH, encoding="utf-8")).items()
        }
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        self.net = EmbeddingClassifierHead(num_classes=checkpoint["num_classes"]).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        self._feat_mean = checkpoint["feat_mean"]
        self._feat_std = checkpoint["feat_std"]

        if os.path.exists(config.CENTROIDS_PATH):
            self.centroids = json.load(open(config.CENTROIDS_PATH, encoding="utf-8"))
        return True

    @property
    def is_trained(self):
        return self.net is not None