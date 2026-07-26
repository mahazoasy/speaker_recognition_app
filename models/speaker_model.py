"""
models/speaker_model.py
--------------------------
Encapsule le cycle de vie complet du modèle de reconnaissance du
locuteur : préparation du dataset, entraînement, sauvegarde/chargement,
identification (qui parle ?) et vérification (est-ce bien X ?).
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import config
from models.neural_network import SpeakerEmbeddingNet
from utils.feature_extractor import features_from_file


class VoiceDataset(Dataset):
    """Dataset PyTorch construit à partir des enregistrements en base."""

    def __init__(self, records, label_to_idx):
        # records: liste de dicts {filepath, name}
        self.records = records
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        feats = features_from_file(rec["filepath"])
        label = self.label_to_idx[rec["name"]]
        return torch.from_numpy(feats), label


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
        self.net: SpeakerEmbeddingNet | None = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.centroids = {}  # name -> embedding moyen (liste de floats)

    # ------------------------------------------------------------------
    # Préparation
    # ------------------------------------------------------------------
    def _build_label_maps(self, records):
        names = sorted({r["name"] for r in records})
        self.label_to_idx = {name: i for i, name in enumerate(names)}
        self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}

    def _ensure_net(self, num_classes):
        if self.net is None:
            self.net = SpeakerEmbeddingNet(num_classes=num_classes).to(self.device)
        elif self.net.num_classes != num_classes:
            self.net.resize_classifier(num_classes)
            self.net.to(self.device)

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    def train(self, records, epochs=config.DEFAULT_EPOCHS,
              batch_size=config.DEFAULT_BATCH_SIZE, lr=config.DEFAULT_LR,
              epoch_callback=None, stop_flag=None):
        """
        records : liste de {filepath, name} provenant de la base de données.
        epoch_callback(epoch, loss, acc, val_loss, val_acc) : appelé à chaque époque
        stop_flag : objet avec attribut .stopped (bool) pour interrompre proprement
        """
        if len(records) < 2:
            raise ValueError("Il faut au moins 2 enregistrements pour entraîner le modèle.")

        self._build_label_maps(records)
        num_classes = len(self.label_to_idx)
        if num_classes < 2:
            raise ValueError("Il faut au moins 2 locuteurs différents pour entraîner le modèle.")

        self._ensure_net(num_classes)

        full_dataset = VoiceDataset(records, self.label_to_idx)

        # Split train/validation (80/20), avec garde-fou pour petits datasets
        n_val = max(1, int(0.2 * len(full_dataset))) if len(full_dataset) >= 5 else 0
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

        history = []
        for epoch in range(1, epochs + 1):
            if stop_flag is not None and getattr(stop_flag, "stopped", False):
                break

            self.net.train()
            total_loss, correct, total = 0.0, 0, 0
            for feats, labels in train_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                _, logits = self.net(feats)
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

        # Recalcule les empreintes moyennes (centroïdes) par locuteur
        self._compute_centroids(full_dataset)
        self.save()
        return history

    def _evaluate(self, loader, criterion):
        self.net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for feats, labels in loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                _, logits = self.net(feats)
                loss = criterion(logits, labels)
                total_loss += loss.item() * feats.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += feats.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    def _compute_centroids(self, dataset):
        self.net.eval()
        sums = {name: None for name in self.label_to_idx}
        counts = {name: 0 for name in self.label_to_idx}
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        with torch.no_grad():
            idx = 0
            for feats, labels in loader:
                feats = feats.to(self.device)
                emb, _ = self.net(feats)
                emb = emb.cpu().numpy()
                for i, label_idx in enumerate(labels.numpy()):
                    name = self.idx_to_label[int(label_idx)]
                    if sums[name] is None:
                        sums[name] = emb[i].copy()
                    else:
                        sums[name] += emb[i]
                    counts[name] += 1
        self.centroids = {
            name: (sums[name] / counts[name]).tolist()
            for name in sums if sums[name] is not None
        }

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def _embed(self, audio: np.ndarray):
        from utils.feature_extractor import extract_features
        feats = extract_features(audio)
        tensor = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            embedding, logits = self.net(tensor)
        return embedding.cpu().numpy()[0], logits.cpu().numpy()[0]

    def predict(self, audio: np.ndarray):
        """Identification en 'closed set' : renvoie {nom: probabilité}."""
        if self.net is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné.")
        embedding, logits = self._embed(audio)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        result = {self.idx_to_label[i]: float(p) for i, p in enumerate(probs)}
        best_name = max(result, key=result.get)
        return best_name, result[best_name], result, embedding

    def verify(self, audio: np.ndarray, claimed_name: str):
        """Vérification 1:1 via similarité cosinus à l'empreinte moyenne."""
        if claimed_name not in self.centroids:
            raise ValueError(f"Aucune empreinte enregistrée pour '{claimed_name}'.")
        embedding, _ = self._embed(audio)
        centroid = np.array(self.centroids[claimed_name])
        sim = float(
            np.dot(embedding, centroid) /
            (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8)
        )
        # Ramène la similarité cosinus [-1, 1] vers une pseudo-probabilité [0, 1]
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
        }, config.MODEL_PATH)
        with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.idx_to_label, f, ensure_ascii=False, indent=2)
        with open(config.CENTROIDS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.centroids, f, ensure_ascii=False, indent=2)

    def load(self):
        if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.LABELS_PATH)):
            return False
        checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
        self.idx_to_label = {
            int(k): v for k, v in json.load(open(config.LABELS_PATH, encoding="utf-8")).items()
        }
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        self.net = SpeakerEmbeddingNet(num_classes=checkpoint["num_classes"]).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

        if os.path.exists(config.CENTROIDS_PATH):
            self.centroids = json.load(open(config.CENTROIDS_PATH, encoding="utf-8"))
        return True

    @property
    def is_trained(self):
        return self.net is not None
