# filepath: train_xgboost.py
"""
Classification des vocalisations de baleines — XGBoost avec features acoustiques
=================================================================================
Au lieu d'aplatir les pixels bruts (trop bruité, pas de sens physique),
on extrait des descripteurs acoustiques qui capturent la forme des vocalisations :
  - Énergie et statistiques spectrales
  - Fréquence centroïde et sa trajectoire temporelle (pente = signe distinctif)
  - Bande passante
  - Contraste spectral
  - Moments statistiques du patch
"""

import logging
from pathlib import Path

import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path(r"C:\ENSTA\2A\S4\Machine_learning\Projet\dataset_prepared")
IMG_SIZE    = (128, 128)
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# EXTRACTION DE FEATURES ACOUSTIQUES
# ─────────────────────────────────────────────

def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extrait ~60 features acoustiques depuis une imagette de spectrogramme.
    L'image est en niveaux de gris, shape (H, W), valeurs [0, 255].
    H = axe fréquentiel (bas = basse fréq), W = axe temporel.
    """
    img_f = img.astype(np.float32) / 255.0
    H, W  = img_f.shape
    features = []

    # ── 1. Statistiques globales du patch ──────────────────────────────
    features += [
        img_f.mean(),
        img_f.std(),
        np.percentile(img_f, 25),
        np.percentile(img_f, 75),
        img_f.max(),
        img_f.min(),
    ]

    # ── 2. Profil fréquentiel moyen ─────────────────────────────────────
    freq_profile = img_f.mean(axis=1)
    features += [
        freq_profile.mean(),
        freq_profile.std(),
        float(np.argmax(freq_profile)) / H,
        float(np.argmin(freq_profile)) / H,
    ]
    freq_idx   = np.arange(H)
    centroid_f = np.sum(freq_idx * freq_profile) / (freq_profile.sum() + 1e-8)
    features.append(centroid_f / H)
    bw = np.sqrt(np.sum(((freq_idx - centroid_f) ** 2) * freq_profile) /
                 (freq_profile.sum() + 1e-8))
    features.append(bw / H)

    # ── 3. Profil temporel moyen ────────────────────────────────────────
    time_profile = img_f.mean(axis=0)
    features += [
        time_profile.mean(),
        time_profile.std(),
        float(np.argmax(time_profile)) / W,
    ]

    # ── 4. Trajectoire du centroïde fréquentiel dans le temps ──────────
    # Feature clé : BmA est horizontal, BmZ descend lentement, BpD vite
    centroids_t = []
    for t in range(W):
        col = img_f[:, t]
        c   = np.sum(freq_idx * col) / (col.sum() + 1e-8)
        centroids_t.append(c / H)
    centroids_t = np.array(centroids_t)

    x     = np.linspace(0, 1, W)
    slope = np.polyfit(x, centroids_t, 1)[0]
    features += [
        centroids_t.mean(),
        centroids_t.std(),
        slope,
        centroids_t.max() - centroids_t.min(),
    ]

    # ── 5. Contraste spectral haut/bas ──────────────────────────────────
    mid = H // 2
    energy_low  = img_f[:mid, :].mean()
    energy_high = img_f[mid:, :].mean()
    features += [energy_low, energy_high, energy_high - energy_low]

    # ── 6. Gradients (texture / netteté des transitions) ────────────────
    grad_x = np.diff(img_f, axis=1)
    grad_y = np.diff(img_f, axis=0)
    features += [
        np.abs(grad_x).mean(),
        np.abs(grad_x).std(),
        np.abs(grad_y).mean(),
        np.abs(grad_y).std(),
    ]

    # ── 7. Énergie par blocs 4×4 ────────────────────────────────────────
    block_h = H // 4
    block_w = W // 4
    for i in range(4):
        for j in range(4):
            block = img_f[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            features.append(block.mean())

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_split(split: str):
    split_dir  = DATASET_DIR / split
    X, y       = [], []
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
            X.append(extract_features(img))
            y.append(label)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def main():
    logger.info("Chargement train...")
    X_train, y_train = load_split("train")
    logger.info("Chargement validation...")
    X_val, y_val = load_split("validation")

    logger.info("Features shape — train: %s | val: %s", X_train.shape, X_val.shape)

    # Encodage labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val]))
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    classes     = list(le.classes_)
    logger.info("Classes : %s", classes)

    # SMOTE
    logger.info("SMOTE...")
    _, counts = np.unique(y_train_enc, return_counts=True)
    logger.info("Avant SMOTE: %s", counts)
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train_enc)
    _, counts_sm = np.unique(y_train_sm, return_counts=True)
    logger.info("Après SMOTE: %s", counts_sm)

    # Standardisation
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_val_sc   = scaler.transform(X_val)

    # XGBoost
    logger.info("Entraînement XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators          = 150,
        max_depth             = 8,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 30,
        random_state          = RANDOM_SEED,
        n_jobs                = -1,
    )
    model.fit(
        X_train_sc, y_train_sm,
        eval_set=[(X_val_sc, y_val_enc)],
        verbose=50,
    )

    # Évaluation
    y_pred = le.inverse_transform(model.predict(X_val_sc))
    acc    = accuracy_score(y_val, y_pred)
    logger.info("Accuracy sur validation : %.2f%%", acc * 100)

    print("\n── Rapport de classification ──")
    print(classification_report(y_val, y_pred, target_names=classes))

    # Noms des features
    feat_names = (
        ["mean", "std", "p25", "p75", "max", "min",
         "freq_mean", "freq_std", "freq_argmax", "freq_argmin",
         "centroid_f", "bandwidth",
         "time_mean", "time_std", "time_argmax",
         "centroid_t_mean", "centroid_t_std", "slope", "excursion",
         "energy_low", "energy_high", "contrast",
         "grad_x_mean", "grad_x_std", "grad_y_mean", "grad_y_std"]
        + [f"block_{i}{j}" for i in range(4) for j in range(4)]
    )
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:15]

    # Figures
    cm  = confusion_matrix(y_val, y_pred, labels=classes)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")
    axes[0].set_title(f"Matrice de confusion — Accuracy {acc*100:.1f}%")

    axes[1].barh([feat_names[i] for i in top_idx[::-1]],
                 importances[top_idx[::-1]])
    axes[1].set_title("Top 15 features les plus importantes")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("xgboost_results.png", dpi=150)
    logger.info("Résultats → xgboost_results.png")
    plt.show()


if __name__ == "__main__":
    main()