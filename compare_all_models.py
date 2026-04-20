# filepath: compare_all_models.py

"""
Comparaison de tous les classifieurs — BioDCASE
================================================
Ce script entraîne les 5 modèles du projet sur le même split
train/test et produit :
  - Un tableau récapitulatif (accuracy + F1 macro + F1 weighted)
  - Un graphique en barres comparatif
  - Les matrices de confusion côte à côte

Modèles comparés :
  1. Naive Bayes (Gaussian)
  2. XGBoost
  3. SVM (RBF)
  4. Random Forest
  5. MLP
"""

import logging
import time
from pathlib import Path

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample as sk_resample
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

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
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img.flatten().astype(np.float32) / 255.0)
            y.append(label)

    logger.info("  → %d images chargées", len(y))
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# PREPROCESSING COMMUN
# ─────────────────────────────────────────────

def preprocess(X_train, y_train_enc, X_test, use_smote=True):
    """Normalisation StandardScaler + PCA + SMOTE optionnel."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info("  Variance PCA expliquée : %.1f%%", explained)

    if use_smote:
        smote = SMOTE(sampling_strategy="auto", k_neighbors=5,
                      random_state=RANDOM_SEED, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X_train_pca, y_train_enc)
    else:
        # Simple oversampling pour Naive Bayes (SMOTE peut dégénérer sur GNB)
        classes   = np.unique(y_train_enc)
        max_size  = max(np.sum(y_train_enc == c) for c in classes)
        X_parts, y_parts = [], []
        for c in classes:
            mask = y_train_enc == c
            Xc, yc = sk_resample(X_train_pca[mask], y_train_enc[mask],
                                  replace=True, n_samples=max_size,
                                  random_state=RANDOM_SEED)
            X_parts.append(Xc); y_parts.append(yc)
        X_res = np.vstack(X_parts)
        y_res = np.hstack(y_parts)

    return X_res, y_res, X_test_pca


# ─────────────────────────────────────────────
# DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────

def get_models():
    return {
        "Naive Bayes": (
            GaussianNB(),
            False   # use_smote = False → simple oversampling
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1,
            ),
            True
        ),
        "SVM (RBF)": (
            SVC(C=10, kernel="rbf", gamma="scale",
                decision_function_shape="ovr", random_state=RANDOM_SEED),
            True
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
            ),
            True
        ),
        "MLP": (
            MLPClassifier(
                hidden_layer_sizes=(512, 256, 128), activation="relu",
                solver="adam", alpha=1e-4, batch_size=128,
                learning_rate="adaptive", learning_rate_init=1e-3,
                max_iter=200, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=15, random_state=RANDOM_SEED,
            ),
            True
        ),
    }


# ─────────────────────────────────────────────
# ENTRAÎNEMENT + ÉVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(models, X_train, y_train_enc, X_test, y_test_enc,
                       y_test, classes, le):
    results = []
    cms     = {}

    for name, (model, use_smote) in models.items():
        logger.info("\n════════════════════════════════")
        logger.info("  Modèle : %s", name)
        logger.info("════════════════════════════════")

        X_tr, y_tr, X_te = preprocess(X_train, y_train_enc, X_test,
                                       use_smote=use_smote)

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0
        logger.info("  Temps d'entraînement : %.1f s", train_time)

        y_pred_enc = model.predict(X_te)
        y_pred     = le.inverse_transform(y_pred_enc)

        acc        = accuracy_score(y_test, y_pred)
        f1_macro   = f1_score(y_test, y_pred, average="macro",   zero_division=0)
        f1_weighted= f1_score(y_test, y_pred, average="weighted",zero_division=0)

        logger.info("  Accuracy  : %.2f%%", acc * 100)
        logger.info("  F1 macro  : %.4f",  f1_macro)
        logger.info("  F1 weighted: %.4f", f1_weighted)

        print(f"\n── {name} — Rapport détaillé ──────────────────────────────")
        print(classification_report(y_test, y_pred, target_names=classes,
                                    zero_division=0))

        results.append({
            "Modèle"       : name,
            "Accuracy (%)" : round(acc * 100, 2),
            "F1 Macro"     : round(f1_macro, 4),
            "F1 Weighted"  : round(f1_weighted, 4),
            "Temps (s)"    : round(train_time, 1),
        })

        cms[name] = confusion_matrix(y_test, y_pred, labels=classes)

    return pd.DataFrame(results), cms


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────

def plot_comparison(df_results):
    """Graphique en barres : Accuracy + F1 macro + F1 weighted."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["Accuracy (%)", "F1 Macro", "F1 Weighted"]
    colors  = ["steelblue", "seagreen", "tomato"]

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(df_results["Modèle"], df_results[metric], color=color,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, df_results[metric].max() * 1.15)
        ax.set_xticklabels(df_results["Modèle"], rotation=20, ha="right", fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Comparaison des classifieurs — BioDCASE", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("comparison_models.png", dpi=150, bbox_inches="tight")
    logger.info("Sauvegardé → comparison_models.png")
    plt.show()


def plot_all_confusion_matrices(cms, classes):
    """Affiche les 5 matrices de confusion en grille 2×3."""
    names = list(cms.keys())
    n     = len(names)
    cols  = 3
    rows  = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()

    cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

    for i, name in enumerate(names):
        cm = cms[name]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[i % len(cmaps)],
                    xticklabels=classes, yticklabels=classes, ax=axes[i])
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Classe prédite", fontsize=9)
        axes[i].set_ylabel("Classe réelle", fontsize=9)

    # Masquer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Matrices de confusion — tous les modèles", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("all_confusion_matrices.png", dpi=150, bbox_inches="tight")
    logger.info("Sauvegardé → all_confusion_matrices.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Chargement
    logger.info("Chargement des données...")
    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("test")
    logger.info("Train : %s  |  Test : %s", X_train.shape, X_test.shape)

    # Encodage labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)
    classes     = list(le.classes_)
    logger.info("Classes : %s", classes)

    # Entraînement + évaluation
    models = get_models()
    df_results, cms = train_and_evaluate(
        models, X_train, y_train_enc, X_test, y_test_enc, y_test, classes, le
    )

    # Tableau récapitulatif
    print("\n\n══════════════════════════════════════════════════════")
    print("  TABLEAU RÉCAPITULATIF")
    print("══════════════════════════════════════════════════════")
    print(df_results.to_string(index=False))
    df_results.to_csv("results_summary.csv", index=False)
    logger.info("Tableau sauvegardé → results_summary.csv")

    # Graphiques
    plot_comparison(df_results)
    plot_all_confusion_matrices(cms, classes)

    logger.info("\nTerminé. Fichiers générés :")
    logger.info("  comparison_models.png")
    logger.info("  all_confusion_matrices.png")
    logger.info("  results_summary.csv")


if __name__ == "__main__":
    main()
