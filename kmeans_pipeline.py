"""
Clustering K-Means – BioDCASE : Vocalisations de baleines
==========================================================
Compatible avec le pipeline data_processing.py (scipy, SR=250 Hz).

Pipeline complet :
  1. Chargement des imagettes (cache .npz pour relances rapides)
  2. Extraction de features : HOG + statistiques spectrales
  3. Nettoyage (colonnes constantes) + StandardScaler + PCA
  4. Détermination du k optimal : méthode du coude + silhouette
  5. K-Means final (k-means++, 20 initialisations)
  6. Évaluation non supervisée : silhouette, Davies-Bouldin, Calinski-Harabasz
  7. Évaluation supervisée : pureté, NMI, ARI + matrice de confusion alignée
  8. Visualisations : embedding 2D, distribution par cluster, diagramme silhouette

IMPORTANT : Si vous relancez data_processing.py, supprimez dataset_cache.npz
            avant de relancer ce script (sinon le cache périmé sera utilisé).
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
)
from skimage.feature import hog
import cv2
from tqdm import tqdm

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Pointe vers le dossier produit par data_processing.py
# (OUTPUT_ROOT = BASE_DIR / "dataset_prepared" dans Config)
DATASET_ROOT   = Path(__file__).resolve().parent / "dataset_prepared"
CACHE_PATH     = Path(__file__).resolve().parent / "dataset_cache.npz"

IMG_SIZE       = (128, 128)
RANDOM_STATE   = 42
N_PCA_FEATURES = 100           # dimensions avant K-Means
K_RANGE        = range(2, 15)  # valeurs de k à tester
N_CLUSTERS     = 7             # k final = nombre de classes BioDCASE

# Labels en minuscules — data_processing.py fait label.strip().lower()
CLASSES = ["bma", "bmb", "bmd", "bmz", "bp20", "bp20plus", "bpd"]


# ─────────────────────────────────────────────
# 1. CHARGEMENT AVEC CACHE
# ─────────────────────────────────────────────

def load_images() -> tuple:
    """
    Lit les PNG produits par data_processing.py.
    - 1er appel  : parcourt tous les PNG (lent) puis sauvegarde un cache .npz.
    - Appels suivants : recharge le cache en quelques secondes.

    IMPORTANT : Supprimez CACHE_PATH si vous avez relancé data_processing.py.
    """
    if CACHE_PATH.exists():
        logger.info("Cache trouve -> chargement rapide depuis %s", CACHE_PATH)
        data = np.load(CACHE_PATH, allow_pickle=True)
        images = data["images"]
        labels = data["labels"].astype(str)
        logger.info("  %d images, %d classes : %s",
                    len(images), len(set(labels)), sorted(set(labels)))
        return images, labels

    logger.info("Pas de cache -> lecture des PNG depuis %s ...", DATASET_ROOT)
    images, labels = [], []

    for split in ["train", "validation"]:
        split_path = DATASET_ROOT / split
        if not split_path.exists():
            logger.warning("Split '%s' introuvable, ignore.", split)
            continue
        for label_dir in sorted(split_path.iterdir()):
            if not label_dir.is_dir():
                continue
            png_files = list(label_dir.glob("*.png"))
            for p in tqdm(png_files, desc=f"{split}/{label_dir.name}", leave=False):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(cv2.resize(img, IMG_SIZE))
                labels.append(label_dir.name)

    if not images:
        raise FileNotFoundError(
            f"Aucune image trouvee dans {DATASET_ROOT}.\n"
            "Lancez d'abord data_processing.py."
        )

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=str)
    np.savez_compressed(CACHE_PATH, images=X, labels=y)
    logger.info("Cache sauvegarde : %s  (%d images, %d classes)",
                CACHE_PATH, len(X), len(set(y)))
    return X, y


# ─────────────────────────────────────────────
# 2. EXTRACTION DE FEATURES
# ─────────────────────────────────────────────

def _hog_features(images: np.ndarray) -> np.ndarray:
    """
    HOG (Histogram of Oriented Gradients).
    Capture les contours et gradients des spectrogrammes :
    bien adapte aux formes frequentielles des vocalisations.
    """
    return np.array([
        hog(img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys")
        for img in tqdm(images, desc="HOG features", leave=False)
    ])


def _stat_features(images: np.ndarray) -> np.ndarray:
    """
    Statistiques spectrales par ligne (axe frequence) et colonne (axe temps) :
    moyenne, ecart-type, skewness, kurtosis.
    Les NaN produits sur des lignes/colonnes constantes sont remplaces par 0.
    """
    from scipy.stats import skew, kurtosis as scipy_kurtosis

    feats = []
    for img in tqdm(images, desc="Stats features", leave=False):
        f = img.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat = np.concatenate([
                f.mean(axis=1),
                f.std(axis=1),
                np.nan_to_num(skew(f, axis=1),           nan=0.0),
                np.nan_to_num(scipy_kurtosis(f, axis=1), nan=0.0),
                f.mean(axis=0),
                f.std(axis=0),
                np.nan_to_num(skew(f, axis=0),           nan=0.0),
                np.nan_to_num(scipy_kurtosis(f, axis=0), nan=0.0),
            ])
        feats.append(feat)
    return np.array(feats)


def _remove_constant_cols(X: np.ndarray) -> tuple:
    """Supprime les colonnes de variance nulle (std < 1e-8)."""
    mask = X.std(axis=0) > 1e-8
    n_removed = int((~mask).sum())
    if n_removed:
        logger.info("Colonnes constantes supprimees : %d / %d", n_removed, X.shape[1])
    return X[:, mask], mask


def extract_features(images: np.ndarray) -> tuple:
    """
    Retourne (X_clean, mask_colonnes).
    X_clean = HOG || stats, sans colonnes constantes.
    """
    X = np.concatenate([_hog_features(images), _stat_features(images)], axis=1)
    X, mask = _remove_constant_cols(X)
    logger.info("Feature matrix finale : %s", X.shape)
    return X, mask


# ─────────────────────────────────────────────
# 3. PRETRAITEMENT : StandardScaler + PCA
# ─────────────────────────────────────────────

def preprocess(X: np.ndarray) -> tuple:
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    pca   = PCA(n_components=N_PCA_FEATURES, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sc)

    var_cumul = pca.explained_variance_ratio_.cumsum()[-1]
    logger.info("PCA %d composantes -> %.1f %% de variance expliquee",
                N_PCA_FEATURES, var_cumul * 100)
    return X_pca, scaler, pca


# ─────────────────────────────────────────────
# 4. SELECTION DU K OPTIMAL
# ─────────────────────────────────────────────

def find_optimal_k(X_pca: np.ndarray,
                   k_range: range = K_RANGE,
                   save_path: str = "elbow_silhouette_kmeans.png"):
    """
    Pour chaque k dans k_range, calcule :
      - L'inertie intra-cluster (critere du coude / WCSS)
      - Le score de silhouette moyen

    Les deux courbes sont tracees cote a cote.
    La ligne rouge indique N_CLUSTERS (le k retenu).
    """
    inertias, silhouettes = [], []

    for k in tqdm(k_range, desc="Selection k"):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=RANDOM_STATE)
        lbls = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca, lbls,
                               sample_size=min(3000, len(X_pca)),
                               random_state=RANDOM_STATE)
        silhouettes.append(sil)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(k_range), inertias, "bo-", lw=2)
    ax1.axvline(N_CLUSTERS, color="red", ls="--", label=f"k={N_CLUSTERS} retenu")
    ax1.set_xlabel("k (nombre de clusters)")
    ax1.set_ylabel("Inertie intra-cluster (WCSS)")
    ax1.set_title("Methode du coude")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(list(k_range), silhouettes, "go-", lw=2)
    ax2.axvline(N_CLUSTERS, color="red", ls="--", label=f"k={N_CLUSTERS} retenu")
    ax2.set_xlabel("k (nombre de clusters)")
    ax2.set_ylabel("Score de silhouette moyen")
    ax2.set_title("Score de silhouette vs k")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("K-Means - Selection du nombre de clusters k",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Graphique sauvegarde : %s", save_path)

    best_sil_k = list(k_range)[int(np.argmax(silhouettes))]
    logger.info("k optimal selon silhouette : %d  (retenu : %d)", best_sil_k, N_CLUSTERS)


# ─────────────────────────────────────────────
# 5. K-MEANS FINAL
# ─────────────────────────────────────────────

def run_kmeans(X_pca: np.ndarray, n_clusters: int = N_CLUSTERS) -> tuple:
    """
    K-Means avec :
    - initialisation k-means++ (evite les mauvais minima locaux)
    - 20 restarts independants (on garde le meilleur)
    """
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    cluster_labels = km.fit_predict(X_pca)
    logger.info("K-Means converge en %d iteration(s) | inertie finale = %.2f",
                km.n_iter_, km.inertia_)
    return km, cluster_labels


# ─────────────────────────────────────────────
# 6. METRIQUES NON SUPERVISEES
# ─────────────────────────────────────────────

def evaluate_unsupervised(X_pca: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """
    Trois indices internes (ne necessitent pas les vrais labels) :

    - Silhouette  in [-1, +1]  -> proche de +1 = clusters bien separes et denses
    - Davies-Bouldin >= 0      -> proche de 0  = clusters compacts et bien separes
    - Calinski-Harabasz >= 0   -> plus grand   = meilleure separation inter/intra
    """
    sil = silhouette_score(X_pca, cluster_labels,
                           sample_size=min(5000, len(X_pca)),
                           random_state=RANDOM_STATE)
    db  = davies_bouldin_score(X_pca, cluster_labels)
    ch  = calinski_harabasz_score(X_pca, cluster_labels)

    print("\n" + "="*50)
    print("  METRIQUES NON SUPERVISEES")
    print("="*50)
    print(f"  Score de silhouette      : {sil:+.4f}")
    print(f"  Indice Davies-Bouldin    : {db:.4f}")
    print(f"  Indice Calinski-Harabasz : {ch:.1f}")
    print("-"*50)
    print("  Silhouette in [-1,+1]  -> +1 ideal")
    print("  Davies-Bouldin -> 0 ideal (grand = pire)")
    print("  Calinski-Harabasz -> grand ideal")
    print("="*50 + "\n")

    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


# ─────────────────────────────────────────────
# 7. METRIQUES SUPERVISEES
# ─────────────────────────────────────────────

def _cluster_purity(y_true_int: np.ndarray, y_pred_int: np.ndarray) -> float:
    cm = confusion_matrix(y_true_int, y_pred_int)
    return cm.max(axis=0).sum() / cm.sum()


def _align_clusters(cluster_labels: np.ndarray,
                    true_labels_int: np.ndarray,
                    n_clusters: int) -> np.ndarray:
    """
    Algorithme hongrois : trouve le mapping cluster->classe qui maximise
    le nombre de predictions correctes.
    """
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


def evaluate_supervised(cluster_labels: np.ndarray,
                        true_labels_int: np.ndarray,
                        class_names: list,
                        n_clusters: int = N_CLUSTERS) -> tuple:
    """
    Trois metriques externes (comparaison avec les vrais labels) :

    - Purete    in [0, 1]   -> fraction d'images bien assignees
    - NMI       in [0, 1]   -> Information Mutuelle Normalisee
    - ARI       in [-1, +1] -> Adjusted Rand Index (0 = aleatoire, +1 = parfait)
    """
    nmi = normalized_mutual_info_score(true_labels_int, cluster_labels)
    ari = adjusted_rand_score(true_labels_int, cluster_labels)
    pur = _cluster_purity(true_labels_int, cluster_labels)

    print("\n" + "="*55)
    print("  METRIQUES SUPERVISEES  (clustering vs vrais labels)")
    print("="*55)
    print(f"  Purete (Purity)                 : {pur:.4f}")
    print(f"  Information Mutuelle Normalisee : {nmi:.4f}")
    print(f"  Adjusted Rand Index (ARI)       : {ari:+.4f}")
    print("-"*55)
    print("  Purete in [0,1]   -> 1 ideal")
    print("  NMI    in [0,1]   -> 1 ideal")
    print("  ARI    in [-1,+1] -> +1 ideal, 0 = aleatoire")
    print("="*55 + "\n")

    aligned_pred = _align_clusters(cluster_labels, true_labels_int, n_clusters)
    return {"purity": pur, "nmi": nmi, "ari": ari}, aligned_pred


# ─────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true_str: np.ndarray,
                          aligned_pred_int: np.ndarray,
                          le: LabelEncoder,
                          class_names: list,
                          save_path: str = "cm_kmeans.png"):
    """
    Matrice de confusion apres alignement optimal des clusters sur les classes.
    Gauche : comptages absolus  |  Droite : rappel normalise par ligne (TVP).
    """
    y_pred_str = le.inverse_transform(aligned_pred_int)
    cm      = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Comptages absolus")
    axes[0].set_xlabel("Cluster aligne -> classe predite")
    axes[0].set_ylabel("Vraie classe")

    sns.heatmap(np.round(cm_norm, 2), annot=True, fmt=".2f",
                cmap="Blues", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Normalisee par ligne (TVP / rappel par classe)")
    axes[1].set_xlabel("Cluster aligne -> classe predite")
    axes[1].set_ylabel("Vraie classe")

    plt.suptitle("K-Means - Matrice de confusion (clusters alignes sur classes BioDCASE)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Matrice sauvegardee : %s", save_path)


def plot_cluster_composition(cluster_labels: np.ndarray,
                              true_labels_str: np.ndarray,
                              class_names: list,
                              n_clusters: int,
                              save_path: str = "cluster_dist_kmeans.png"):
    """
    Un histogramme par cluster montrant la distribution des vraies classes.
    La purete locale est indiquee dans le titre de chaque sous-graphe.
    """
    ncols = 3
    nrows = (n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    palette   = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    color_map = {c: palette[i] for i, c in enumerate(class_names)}

    for k in range(n_clusters):
        mask = cluster_labels == k
        ax   = axes[k]
        if not mask.any():
            ax.set_visible(False)
            continue
        subset = true_labels_str[mask]
        unique, counts = np.unique(subset, return_counts=True)
        purity_k = counts.max() / counts.sum()
        ax.bar(unique, counts,
               color=[color_map.get(c, "gray") for c in unique],
               edgecolor="white")
        ax.set_title(f"Cluster {k}  |  n={mask.sum()}  |  purete={purity_k:.2f}",
                     fontsize=10)
        ax.set_xlabel("Classe reelle")
        ax.set_ylabel("Nombre d'images")
        ax.tick_params(axis="x", rotation=40)

    for j in range(n_clusters, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("K-Means - Composition des clusters (vraies classes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Distribution par cluster sauvegardee : %s", save_path)


def plot_2d_embedding(X_pca: np.ndarray,
                      cluster_labels: np.ndarray,
                      true_labels_str: np.ndarray,
                      class_names: list,
                      save_path: str = "embedding_kmeans.png"):
    """
    Projection 2D : UMAP si installe (pip install umap-learn), sinon PCA 2D.
    Gauche : couleur = cluster K-Means  |  Droite : couleur = vraie classe.
    """
    if HAS_UMAP:
        logger.info("Calcul UMAP 2D...")
        reducer = umap_lib.UMAP(n_components=2, random_state=RANDOM_STATE,
                                n_neighbors=30, min_dist=0.1)
        X_2d   = reducer.fit_transform(X_pca)
        method = "UMAP"
    else:
        logger.info("umap-learn absent -> PCA 2D.")
        X_2d   = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_pca)
        method = "PCA"

    n_k    = max(cluster_labels) + 1
    c_clus = plt.cm.tab10(np.linspace(0, 1, n_k))
    c_cls  = plt.cm.Set1(np.linspace(0, 1, len(class_names)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for k in range(n_k):
        mask = cluster_labels == k
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    color=c_clus[k], s=6, alpha=0.5, label=f"Cluster {k}")
    ax1.set_title(f"Clusters K-Means ({method})")
    ax1.legend(markerscale=3, fontsize=8)
    ax1.axis("off")

    for i, cls in enumerate(class_names):
        mask = true_labels_str == cls
        if not mask.any():
            continue
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    color=c_cls[i], s=6, alpha=0.5, label=cls)
    ax2.set_title(f"Vraies classes BioDCASE ({method})")
    ax2.legend(markerscale=3, fontsize=8)
    ax2.axis("off")

    plt.suptitle(f"K-Means - Espace de representation 2D ({method})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Embedding sauvegarde : %s", save_path)


def plot_silhouette_diagram(X_pca: np.ndarray,
                             cluster_labels: np.ndarray,
                             save_path: str = "silhouette_detail_kmeans.png"):
    """
    Diagramme de silhouette par cluster.
    Chaque bande = un cluster ; largeur = indice de cohesion interne.
    Ligne rouge = silhouette moyenne globale.
    Un cluster dont la bande est entierement a gauche de la ligne rouge
    est sous-performant (mal separe ou trop heterogene).
    """
    sil_vals   = silhouette_samples(X_pca, cluster_labels)
    n_clusters = len(np.unique(cluster_labels))
    colors     = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10

    for k, col in zip(range(n_clusters), colors):
        vals    = np.sort(sil_vals[cluster_labels == k])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=col, edgecolor=col, alpha=0.75)
        ax.text(-0.06, y_lower + 0.45 * len(vals), str(k), fontsize=9)
        y_lower = y_upper + 10

    mean_sil = sil_vals.mean()
    ax.axvline(mean_sil, color="red", ls="--",
               label=f"Silhouette moy. = {mean_sil:.3f}")
    ax.set_xlabel("Valeur de silhouette")
    ax.set_ylabel("Cluster (trie par valeur)")
    ax.set_title("Diagramme de silhouette par cluster")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Diagramme de silhouette sauvegarde : %s", save_path)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Chargement
    logger.info("=== 1. Chargement des images ===")
    images, labels_str = load_images()

    le = LabelEncoder()
    present_classes = sorted({c for c in CLASSES if c in labels_str})
    le.fit(present_classes)
    class_names = le.classes_.tolist()
    labels_int  = le.transform(labels_str)
    logger.info("Classes presentes : %s", class_names)

    # 2. Features
    logger.info("=== 2. Extraction des features ===")
    X, _feat_mask = extract_features(images)

    # 3. Pretraitement
    logger.info("=== 3. Pretraitement (StandardScaler + PCA) ===")
    X_pca, _scaler, _pca = preprocess(X)

    # 4. Selection de k
    logger.info("=== 4. Selection du k optimal ===")
    find_optimal_k(X_pca)

    # 5. K-Means final
    logger.info("=== 5. K-Means avec k=%d ===", N_CLUSTERS)
    km, cluster_labels = run_kmeans(X_pca, n_clusters=N_CLUSTERS)

    # 6. Evaluation non supervisee
    logger.info("=== 6. Metriques non supervisees ===")
    evaluate_unsupervised(X_pca, cluster_labels)

    # 7. Evaluation supervisee
    logger.info("=== 7. Metriques supervisees ===")
    _metrics, aligned_pred = evaluate_supervised(
        cluster_labels, labels_int, class_names
    )

    # 8. Visualisations
    logger.info("=== 8. Visualisations ===")
    plot_confusion_matrix(labels_str, aligned_pred, le, class_names)
    plot_cluster_composition(cluster_labels, labels_str, class_names, N_CLUSTERS)
    plot_2d_embedding(X_pca, cluster_labels, labels_str, class_names)
    plot_silhouette_diagram(X_pca, cluster_labels)

    logger.info("=== Termine ===")


if __name__ == "__main__":
    main()
