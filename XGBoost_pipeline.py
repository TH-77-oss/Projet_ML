# filepath: train_xgboost.py

"""
Classification des vocalisations de baleines avec XGBoost
=========================================================
Pipeline :
  1. Chargement des imagettes (train + validation)
  2. Extraction de features : aplatissement de l'image (pixel brut)
  3. Réduction dimensionnelle : PCA (sinon 16384 features → trop lent)
  4. Entraînement XGBoost
  5. Évaluation : accuracy, F1, matrice de confusion
"""

import logging
from pathlib import Path

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path(r"C:\ENSTA\2A\S4\Machine_learning\Projet\dataset_prepared")
IMG_SIZE    = (128, 128)
N_PCA       = 200      # Nombre de composantes PCA conservées
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# 1. CHARGEMENT DES IMAGES
# ─────────────────────────────────────────────

def load_split(split: str):
    """
    Charge toutes les images d'un split (train ou validation).
    Retourne X (n_samples, H*W) et y (labels string).
    """
    split_dir = DATASET_DIR / split
    X, y = [], []

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    logger.info("Split '%s' : %d classes trouvées", split, len(class_dirs))

    for class_dir in class_dirs:
        label = class_dir.name
        images = list(class_dir.glob("*.png"))
        logger.info("  %-12s : %d images", label, len(images))

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)          # sécurité si taille differ
            X.append(img.flatten().astype(np.float32) / 255.0)   # normalisation [0,1]
            y.append(label)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 2. PIPELINE COMPLET
# ─────────────────────────────────────────────

def main():
    # ── Chargement ──────────────────────────────────────────────────────
    logger.info("Chargement du split train...")
    X_train, y_train = load_split("train")

    logger.info("Chargement du split validation...")
    X_val, y_val = load_split("validation")

    logger.info("Train : %s | Val : %s", X_train.shape, X_val.shape)

    # ── Encodage des labels (string → entier) ───────────────────────────
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val]))
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)

    classes = list(le.classes_)
    logger.info("Classes : %s", classes)

    # ── PCA ─────────────────────────────────────────────────────────────
    # 128×128 = 16 384 features → trop lourd pour XGBoost
    # PCA réduit à N_PCA composantes en gardant le maximum de variance
    logger.info("PCA : réduction %d → %d dimensions...", X_train.shape[1], N_PCA)
    pca = PCA(n_components=N_PCA, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)

    logger.info("SMOTE oversampling...")

    smote = SMOTE(
        sampling_strategy="auto",   # équilibre toutes les classes
        k_neighbors=5,
        random_state=42,
        n_jobs=-1
    )

    X_train_res, y_train_res = smote.fit_resample(X_train_pca, y_train_enc)

    logger.info("Avant SMOTE: %s", np.bincount(y_train_enc))
    logger.info("Après SMOTE: %s", np.bincount(y_train_res))

    explained = pca.explained_variance_ratio_.sum() * 100
    if explained < 0.9:
        logger.warning("Variance trop faible → augmente N_PCA")
    logger.info("Variance expliquée par %d composantes : %.1f%%", N_PCA, explained)

    # ── XGBoost ─────────────────────────────────────────────────────────
    logger.info("Entraînement XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
    )

    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val_pca, y_val_enc)],
        verbose=50,
    )

    # ── Évaluation ──────────────────────────────────────────────────────
    y_pred_enc = model.predict(X_val_pca)
    y_pred     = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_val, y_pred)
    logger.info("Accuracy sur validation : %.2f%%", acc * 100)

    print("\n── Rapport de classification ──────────────────────────────────")
    print(classification_report(y_val, y_pred, target_names=classes))

    # ── Matrice de confusion ─────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"Matrice de confusion XGBoost — Accuracy {acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig("confusion_matrix_xgboost.png", dpi=150)
    logger.info("Matrice de confusion sauvegardée → confusion_matrix_xgboost.png")
    plt.show()


if __name__ == "__main__":
    main()