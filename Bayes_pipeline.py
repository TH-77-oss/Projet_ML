# filepath: train_naive_bayes.py

"""
Classification des vocalisations de baleines avec Naive Bayes
=============================================================
Pipeline :
  1. Chargement des imagettes (train + validation)
  2. Extraction de features : aplatissement
  3. PCA
  4. Naive Bayes (Gaussian)
  5. Évaluation
"""

import logging
from pathlib import Path

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path(r"C:\ENSTA\2A\S4\Machine_learning\Projet\dataset_prepared")
IMG_SIZE    = (128, 128)
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_split(split: str):
    X, y = [], []

    split_dir = DATASET_DIR / split
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    logger.info("Split '%s' : %d classes", split, len(class_dirs))

    for class_dir in class_dirs:
        label = class_dir.name
        images = list(class_dir.glob("*.png"))
        logger.info("  %-12s : %d images", label, len(images))

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            X.append(img.flatten().astype(np.float32) / 255.0)
            y.append(label)

    return np.array(X), np.array(y)

from sklearn.utils import resample

def oversample(X, y):
    X_res, y_res = [], []

    classes = np.unique(y)
    max_size = max(np.sum(y == c) for c in classes)

    for c in classes:
        X_c = X[y == c]
        y_c = y[y == c]

        X_up, y_up = resample(
            X_c, y_c,
            replace=True,
            n_samples=max_size,
            random_state=42
        )

        X_res.append(X_up)
        y_res.append(y_up)

    return np.vstack(X_res), np.hstack(y_res)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("Chargement train...")
    X_train, y_train = load_split("train")

    X_train, y_train = oversample(X_train, y_train)

    logger.info("Chargement validation...")
    X_val, y_val = load_split("validation")

    logger.info("Train : %s | Val : %s", X_train.shape, X_val.shape)

    # Label encoding
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val]))
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)

    classes = list(le.classes_)
    logger.info("Classes : %s", classes)

    # PCA
    #logger.info("PCA %d → %d", X_train.shape[1], N_PCA)
    pca = PCA(n_components=200, svd_solver="randomized", random_state=RANDOM_SEED) # Ajuste n_components pour conserver 95% de la variance
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)

    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info("Variance expliquée : %.1f%%", explained)

    # ── NAIVE BAYES ───────────────────────────
    logger.info("Entraînement Naive Bayes...")
    model = GaussianNB()

    model.fit(X_train_pca, y_train_enc)

    # ── ÉVALUATION ───────────────────────────
    y_pred_enc = model.predict(X_val_pca)
    y_pred     = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_val, y_pred)
    logger.info("Accuracy : %.2f%%", acc * 100)

    print("\n── Rapport de classification ─────────────")
    print(classification_report(y_val, y_pred, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
    )

    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"Naive Bayes — Accuracy {acc*100:.1f}%")

    plt.tight_layout()
    plt.savefig("confusion_matrix_nb.png", dpi=150)
    logger.info("Sauvegardé → confusion_matrix_nb.png")

    plt.show()


if __name__ == "__main__":
    main()