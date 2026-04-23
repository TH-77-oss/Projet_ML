"""
KNN Classifier pour les spectrogrammes du dataset biodcase.
Charge les images depuis dataset_prepared/, entraîne un KNN, et affiche les métriques.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import cv2
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Vérification pour UMAP
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET_DIR = Path(r"C:\Users\anais\Documents\ENSTA_2A\S4_cours\4.1\4.1 Machine Learning\Projet_ML\dataset_prepared")   # généré par data_processing.py
IMG_SIZE    = (128, 128)
K           = 5                          # nombre de voisins
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────
def load_split(split: str):
    """Charge toutes les images d'un split (train / validation) en vecteurs aplatis."""
    split_dir = DATASET_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {split_dir}")

    images, labels = [], []
    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for img_path in label_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img.flatten().astype(np.float32) / 255.0)
            labels.append(label_dir.name)

    logger.info("Split '%s' : %d images, %d classes", split, len(images), len(set(labels)))
    return np.array(images), np.array(labels)


# ──────────────────────────────────────────────
# MÉTRIQUES & VISUALISATION
# ──────────────────────────────────────────────
def print_metrics(y_true, y_pred, class_names):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1-score  : {f1:.4f}  (weighted)")
    print("=" * 50)
    print("\nRapport détaillé par classe :")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(f"Matrice de confusion — KNN (k={K})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    logger.info("Matrice de confusion sauvegardée → confusion_matrix.png")
    plt.show()


def plot_per_class_f1(y_true, y_pred, class_names):
    f1_scores = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), 4))
    ax.bar(class_names, f1_scores, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Classe")
    ax.set_ylabel("F1-score")
    ax.set_title(f"F1-score par classe — KNN (k={K})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("f1_per_class.png", dpi=150)
    logger.info("F1 par classe sauvegardé → f1_per_class.png")
    plt.show()

def plot_knn_2d_visualization(X_data, y_true_str, y_pred_str, class_names, save_path="knn_visualization_2d.png"):
    """
    Visualisation 2D pour comparer les classes réelles et les prédictions KNN.
    """
    # 1. Réduction de dimension intermédiaire (PCA) pour la rapidité
    logger.info("Réduction de dimension (PCA puis 2D)...")
    pca_int = PCA(n_components=min(50, X_data.shape[0]), random_state=42)
    X_reduced = pca_int.fit_transform(X_data)

    # 2. Passage en 2D (UMAP ou PCA)
    if HAS_UMAP:
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_2d = reducer.fit_transform(X_reduced)
        method = "UMAP"
    else:
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X_reduced)
        method = "PCA"

    # 3. Préparation des couleurs
    c_cls = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- GAUCHE : VRAIES CLASSES ---
    for i, cls in enumerate(class_names):
        mask = y_true_str == cls
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], color=c_cls[i], s=10, alpha=0.6, label=cls)
    ax1.set_title(f"Vérité terrain ({method})")
    ax1.legend(markerscale=2, loc='best', fontsize='small')

    # --- DROITE : PRÉDICTIONS ---
    for i, cls in enumerate(class_names):
        mask = y_pred_str == cls
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], color=c_cls[i], s=10, alpha=0.6, label=cls)
    ax2.set_title(f"Prédictions KNN ({method})")
    ax2.legend(markerscale=2, loc='best', fontsize='small')

    plt.suptitle(f"Analyse de la séparation des classes BioDCASE - {method} 2D", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    logger.info(f"Visualisation sauvegardée : {save_path}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    # 1. Chargement
    X_train, y_train_raw = load_split("train")
    X_val,   y_val_raw   = load_split("validation")

    # 2. Encodage des labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_val_raw]))
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    class_names = list(le.classes_)
    logger.info("Classes : %s", class_names)

    # 3. Entraînement KNN
    logger.info("Entraînement KNN (k=%d)…", K)
    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean", n_jobs=-1)
    knn.fit(X_train, y_train)

    # 4. Prédiction
    logger.info("Prédiction sur le split validation…")
    y_pred = knn.predict(X_val)

    '''
    # 5. Métriques
    print_metrics(y_val, y_pred, class_names)

    # 6. Graphiques
    plot_confusion_matrix(y_val, y_pred, class_names)
    plot_per_class_f1(y_val, y_pred, class_names)

    '''

    # 7. Visualisation 2D
    y_val_pred_str = le.inverse_transform(y_pred)
    
    plot_knn_2d_visualization(
        X_data=X_val, 
        y_true_str=y_val_raw, 
        y_pred_str=y_val_pred_str, 
        class_names=class_names
    )

if __name__ == "__main__":
    main()