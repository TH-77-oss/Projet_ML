"""
Comparaison de tous les classifieurs — BioDCASE
================================================
Modèles comparés :
  1. KNN (K Plus Proches Voisins)
  2. Naive Bayes (Gaussian)
  3. XGBoost
  4. SVM (RBF)
  5. Random Forest
  6. MLP

En bonus : évaluation non supervisée par K-Means
(métriques internes + métriques supervisées via alignement hongrois)
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    normalized_mutual_info_score, adjusted_rand_score,
)
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample as sk_resample
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path(r"C:\Users\anais\Documents\ENSTA_2A\S4_cours\4.1\4.1 Machine Learning\Projet_ML\dataset_prepared")   # généré par data_processing.py
IMG_SIZE    = (128, 128)
N_PCA       = 200
RANDOM_SEED = 42
KNN_K       = 5    # nombre de voisins pour le KNN
N_CLUSTERS  = 7    # nombre de clusters pour le K-Means (= nb de classes BioDCASE)


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_split(split: str):
    split_dir = DATASET_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {split_dir}")

    X, y = [], []
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    logger.info("Split '%s' : %d classes", split, len(class_dirs))

    for class_dir in class_dirs:
        for img_path in class_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img.flatten().astype(np.float32) / 255.0)
            y.append(class_dir.name)

    logger.info("  → %d images chargées", len(y))
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# PREPROCESSING COMMUN (StandardScaler + PCA + SMOTE)
# ─────────────────────────────────────────────

def preprocess(X_train, y_train_enc, X_test, use_smote=True):
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
                      random_state=RANDOM_SEED)
        X_res, y_res = smote.fit_resample(X_train_pca, y_train_enc)
    else:
        classes  = np.unique(y_train_enc)
        max_size = max(np.sum(y_train_enc == c) for c in classes)
        X_parts, y_parts = [], []
        for c in classes:
            mask = y_train_enc == c
            Xc, yc = sk_resample(X_train_pca[mask], y_train_enc[mask],
                                  replace=True, n_samples=max_size,
                                  random_state=RANDOM_SEED)
            X_parts.append(Xc); y_parts.append(yc)
        X_res = np.vstack(X_parts)
        y_res = np.hstack(y_parts)

    return X_res, y_res, X_test_pca, scaler, pca


# ─────────────────────────────────────────────
# DÉFINITION DES MODÈLES SUPERVISÉS
# ─────────────────────────────────────────────

def get_models():
    return {
        "KNN": (
            KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean"),
            True,
        ),
        "Naive Bayes": (
            GaussianNB(),
            False,
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=RANDOM_SEED,
            ),
            True,
        ),
        "SVM (RBF)": (
            SVC(C=10, kernel="rbf", gamma="scale",
                decision_function_shape="ovr", random_state=RANDOM_SEED),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                class_weight="balanced", random_state=RANDOM_SEED,
            ),
            True,
        ),
        "MLP": (
            MLPClassifier(
                hidden_layer_sizes=(512, 256, 128), activation="relu",
                solver="adam", alpha=1e-4, batch_size=128,
                learning_rate="adaptive", learning_rate_init=1e-3,
                max_iter=200, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=15, random_state=RANDOM_SEED,
            ),
            True,
        ),
    }


# ─────────────────────────────────────────────
# ENTRAÎNEMENT + ÉVALUATION SUPERVISÉE
# ─────────────────────────────────────────────

def train_and_evaluate(models, X_train, y_train_enc, X_test, y_test_enc,
                       y_test, classes, le):
    results = []
    cms     = {}

    for name, (model, use_smote) in models.items():
        logger.info("\n════════  %s  ════════", name)

        X_tr, y_tr, X_te, _, _ = preprocess(X_train, y_train_enc, X_test,
                                              use_smote=use_smote)

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0
        logger.info("  Temps d'entraînement : %.1f s", train_time)

        y_pred_enc = model.predict(X_te)
        y_pred     = le.inverse_transform(y_pred_enc)

        acc         = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        logger.info("  Accuracy   : %.2f%%", acc * 100)
        logger.info("  F1 macro   : %.4f",  f1_macro)
        logger.info("  F1 weighted: %.4f",  f1_weighted)

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
# K-MEANS : CLUSTERING NON SUPERVISÉ
# ─────────────────────────────────────────────

def _cluster_purity(y_true_int, y_pred_int):
    cm = confusion_matrix(y_true_int, y_pred_int)
    return cm.max(axis=0).sum() / cm.sum()


def _align_clusters(cluster_labels, true_labels_int, n_clusters):
    """Alignement optimal clusters→classes via l'algorithme hongrois."""
    n_classes = len(np.unique(true_labels_int))
    size = max(n_clusters, n_classes)
    cost = np.zeros((size, size), dtype=int)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if not mask.any():
            continue
        for k in range(n_classes):
            cost[c, k] = int((true_labels_int[mask] == k).sum())
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(c), 0) for c in cluster_labels])


def run_kmeans_section(X_train_raw, y_train_raw, le, classes):
    """
    Pipeline K-Means complet sur les données d'entraînement :
      1. PCA partagée avec la branche supervisée
      2. K-Means (k-means++, 20 restarts)
      3. Métriques non supervisées + supervisées
    Retourne un dict de métriques pour l'intégrer au tableau comparatif.
    """
    logger.info("\n════════  K-Means (non supervisé)  ════════")

    # Preprocessing (sans SMOTE — clustering sur données brutes)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train_raw)
    pca    = PCA(n_components=N_PCA, svd_solver="randomized", random_state=RANDOM_SEED)
    X_pca  = pca.fit_transform(X_sc)
    logger.info("  Variance PCA expliquée : %.1f%%",
                pca.explained_variance_ratio_.sum() * 100)

    # K-Means
    t0 = time.time()
    km = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=20,
                max_iter=500, random_state=RANDOM_SEED)
    cluster_labels = km.fit_predict(X_pca)
    train_time = time.time() - t0
    logger.info("  Convergé en %d iter | inertie = %.2f | temps = %.1f s",
                km.n_iter_, km.inertia_, train_time)

    # ── Métriques non supervisées ──
    sil = silhouette_score(X_pca, cluster_labels,
                           sample_size=min(5000, len(X_pca)),
                           random_state=RANDOM_SEED)
    db  = davies_bouldin_score(X_pca, cluster_labels)
    ch  = calinski_harabasz_score(X_pca, cluster_labels)

    print("\n" + "="*50)
    print("  K-MEANS — MÉTRIQUES NON SUPERVISÉES")
    print("="*50)
    print(f"  Silhouette       : {sil:+.4f}  (↑ +1 idéal)")
    print(f"  Davies-Bouldin   : {db:.4f}   (↓ 0 idéal)")
    print(f"  Calinski-Harabasz: {ch:.1f}  (↑ grand idéal)")
    print("="*50)

    # ── Métriques supervisées ──
    true_labels_int = le.transform(y_train_raw)
    nmi = normalized_mutual_info_score(true_labels_int, cluster_labels)
    ari = adjusted_rand_score(true_labels_int, cluster_labels)
    pur = _cluster_purity(true_labels_int, cluster_labels)

    print("\n  K-MEANS — MÉTRIQUES SUPERVISÉES (alignement hongrois)")
    print(f"  Pureté           : {pur:.4f}  (↑ 1 idéal)")
    print(f"  NMI              : {nmi:.4f}  (↑ 1 idéal)")
    print(f"  ARI              : {ari:+.4f}  (↑ +1 idéal, 0=aléatoire)")
    print("="*50)

    # Matrice de confusion alignée
    aligned_pred   = _align_clusters(cluster_labels, true_labels_int, N_CLUSTERS)
    y_pred_classes = le.inverse_transform(aligned_pred)

    print(f"\n── K-Means — Rapport (clusters alignés) ──────────────────────────────")
    print(classification_report(y_train_raw, y_pred_classes,
                                target_names=classes, zero_division=0))

    acc_aligned = accuracy_score(y_train_raw, y_pred_classes)
    f1_macro    = f1_score(y_train_raw, y_pred_classes, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_train_raw, y_pred_classes, average="weighted", zero_division=0)

    cm_kmeans = confusion_matrix(y_train_raw, y_pred_classes, labels=classes)

    return {
        "row": {
            "Modèle"       : f"K-Means (k={N_CLUSTERS})*",
            "Accuracy (%)" : round(acc_aligned * 100, 2),
            "F1 Macro"     : round(f1_macro, 4),
            "F1 Weighted"  : round(f1_weighted, 4),
            "Temps (s)"    : round(train_time, 1),
        },
        "cm": cm_kmeans,
        "sil": sil, "db": db, "ch": ch,
        "purity": pur, "nmi": nmi, "ari": ari,
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────

def plot_comparison(df_results):
    metrics = ["Accuracy (%)", "F1 Macro", "F1 Weighted"]
    colors  = ["steelblue", "seagreen", "tomato"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(df_results["Modèle"], df_results[metric],
                      color=color, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, df_results[metric].max() * 1.15)
        ax.set_xticklabels(df_results["Modèle"], rotation=25, ha="right", fontsize=8)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Comparaison des classifieurs — BioDCASE\n"
                 "* K-Means : métriques calculées sur le train après alignement hongrois",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig("comparison_models.png", dpi=150, bbox_inches="tight")
    logger.info("Sauvegardé → comparison_models.png")
    plt.show()


def plot_all_confusion_matrices(cms, classes):
    names = list(cms.keys())
    cols  = 3
    rows  = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()
    cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlOrBr", "BuGn"]

    for i, name in enumerate(names):
        cm = cms[name]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[i % len(cmaps)],
                    xticklabels=classes, yticklabels=classes, ax=axes[i])
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Classe prédite", fontsize=9)
        axes[i].set_ylabel("Classe réelle", fontsize=9)
        axes[i].tick_params(axis="x", rotation=35)

    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Matrices de confusion — tous les modèles\n"
                 "(K-Means : clusters alignés sur les vraies classes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("all_confusion_matrices.png", dpi=150, bbox_inches="tight")
    logger.info("Sauvegardé → all_confusion_matrices.png")
    plt.show()


def plot_kmeans_extras(sil, db, ch, purity, nmi, ari):
    """Radar / barres des métriques K-Means pour inclusion dans le rapport."""
    metrics = {
        "Silhouette\n(↑ idéal)":      sil,
        "1 - Davies-Bouldin\n(↑ 0)":  max(0, 1 - db),
        "Pureté\n(↑ 1)":             purity,
        "NMI\n(↑ 1)":                nmi,
        "ARI\n(↑ 1)":                max(0, ari),
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(metrics.keys(), metrics.values(), color="mediumpurple",
                  edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"K-Means (k={N_CLUSTERS}) — Métriques internes & externes",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("kmeans_metrics.png", dpi=150, bbox_inches="tight")
    logger.info("Sauvegardé → kmeans_metrics.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Chargement
    logger.info("Chargement des données...")
    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("validation")
    logger.info("Train : %s  |  Test : %s", X_train.shape, X_test.shape)

    # 2. Encodage labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)
    classes     = list(le.classes_)
    logger.info("Classes : %s", classes)

    # 3. Entraînement + évaluation des modèles supervisés
    models = get_models()
    df_results, cms = train_and_evaluate(
        models, X_train, y_train_enc, X_test, y_test_enc, y_test, classes, le
    )

    # 4. K-Means (non supervisé) — sur le train complet
    km_info = run_kmeans_section(X_train, y_train, le, classes)
    df_results = pd.concat(
        [df_results, pd.DataFrame([km_info["row"]])], ignore_index=True
    )
    cms[km_info["row"]["Modèle"]] = km_info["cm"]

    # 5. Tableau récapitulatif
    print("\n\n══════════════════════════════════════════════════════")
    print("  TABLEAU RÉCAPITULATIF")
    print("  (* K-Means : accuracy après alignement hongrois sur le train)")
    print("══════════════════════════════════════════════════════")
    print(df_results.to_string(index=False))
    df_results.to_csv("results_summary.csv", index=False)
    logger.info("Tableau sauvegardé → results_summary.csv")

    # 6. Graphiques
    plot_comparison(df_results)
    plot_all_confusion_matrices(cms, classes)
    plot_kmeans_extras(
        km_info["sil"], km_info["db"], km_info["ch"],
        km_info["purity"], km_info["nmi"], km_info["ari"],
    )

    logger.info("\nFichiers générés :")
    for f in ["comparison_models.png", "all_confusion_matrices.png",
              "kmeans_metrics.png", "results_summary.csv"]:
        logger.info("  %s", f)


if __name__ == "__main__":
    main()