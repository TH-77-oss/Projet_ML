# filepath: mlp_pipeline.py

"""
Classification des vocalisations de baleines avec MLP (réseau de neurones)
===========================================================================
Pipeline :
  1. Chargement des imagettes (train + test)
  2. Aplatissement + normalisation
  3. Réduction dimensionnelle : PCA (200 composantes)
  4. Rééquilibrage : SMOTE
  5. Entraînement MLP (sklearn MLPClassifier)
  6. Évaluation : accuracy, F1, matrice de confusion,
     courbe de loss d'entraînement
"""

import logging
from pathlib import Path

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path(r"C:\ENSTA\2A\S4\Machine_learning\Projet\dataset_prepared")
IMG_SIZE    = (128, 128)
N_PCA       = 200
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_split(split: str):
    split_dir  = DATASET_DIR / split
    X, y = [], []

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    logger.info("Split '%s' : %d classes", split, len(class_dirs))

    for class_dir in class_dirs:
        label  = class_dir.name
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Chargement
    logger.info("Chargement train...")
    X_train, y_train = load_split("train")

    logger.info("Chargement test...")
    X_test, y_test = load_split("test")

    logger.info("Train : %s  |  Test : %s", X_train.shape, X_test.shape)

    # 2. Encodage labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)
    classes     = list(le.classes_)
    logger.info("Classes : %s", classes)

    # 3. Normalisation + PCA
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    logger.info("PCA : %d → %d dimensions...", X_train_sc.shape[1], N_PCA)
    pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info("Variance expliquée : %.1f%%", explained)

    # 4. SMOTE
    logger.info("SMOTE oversampling...")
    smote = SMOTE(sampling_strategy="auto", k_neighbors=5,
                  random_state=RANDOM_SEED, n_jobs=-1)
    X_train_res, y_train_res = smote.fit_resample(X_train_pca, y_train_enc)
    logger.info("Avant SMOTE : %s", np.bincount(y_train_enc))
    logger.info("Après SMOTE : %s", np.bincount(y_train_res))

    # 5. Entraînement MLP
    # Architecture : 200 → 512 → 256 → 128 → n_classes
    # early_stopping évite le surapprentissage
    logger.info("Entraînement MLP...")
    model = MLPClassifier(
        hidden_layer_sizes = (512, 256, 128),
        activation         = "relu",
        solver             = "adam",
        alpha              = 1e-4,          # régularisation L2
        batch_size         = 128,
        learning_rate      = "adaptive",
        learning_rate_init = 1e-3,
        max_iter           = 200,
        early_stopping     = True,
        validation_fraction= 0.1,
        n_iter_no_change   = 15,
        random_state       = RANDOM_SEED,
        verbose            = True,
    )
    model.fit(X_train_res, y_train_res)

    logger.info("Itérations : %d", model.n_iter_)

    # 6. Évaluation
    y_pred_enc = model.predict(X_test_pca)
    y_pred     = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test, y_pred)
    logger.info("Accuracy : %.2f%%", acc * 100)

    print("\n── Rapport de classification ─────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_xlabel("Classe prédite")
    axes[0].set_ylabel("Classe réelle")
    axes[0].set_title(f"MLP — Accuracy {acc*100:.1f}%")

    # Courbe de loss
    axes[1].plot(model.loss_curve_, label="Train loss", color="royalblue")
    if model.best_loss_ is not None:
        axes[1].axhline(y=model.best_loss_, color="red", linestyle="--",
                        label=f"Best loss = {model.best_loss_:.4f}")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Loss (log-loss)")
    axes[1].set_title("Courbe d'apprentissage MLP")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("confusion_matrix_mlp.png", dpi=150)
    logger.info("Sauvegardé → confusion_matrix_mlp.png")
    plt.show()


if __name__ == "__main__":
    main()
