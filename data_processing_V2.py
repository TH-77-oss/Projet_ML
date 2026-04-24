# filepath: data_processing.py

"""
Pipeline de préparation des données BiodCase
=============================================
Fix principal : audio_start est extrait du nom du fichier WAV
et passé explicitement à extract_patch.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

class Config:
    BASE_DIR    = Path(__file__).resolve().parent
    DATA_ROOT   = Path(r"C:\ENSTA\2A\S4\Machine_learning\Projet\biodcase_development_set\biodcase_development_set")
    OUTPUT_ROOT = BASE_DIR / "dataset_prepared"

    SR            = 200
    N_FFT         = 512
    HOP_LENGTH    = 64
    N_MELS        = 128
    F_MIN         = 0
    F_MAX         = None

    IMG_SIZE      = (128, 128)
    RESIZE_METHOD = "resize"
    DYNAMIC_RANGE = 80


cfg = Config()


# ─────────────────────────────────────────────
# UTILITAIRE : timestamp de début du fichier WAV
# ─────────────────────────────────────────────

def get_audio_start(wav_path: Path) -> pd.Timestamp:
    """
    Extrait le timestamp de début depuis le nom du fichier WAV.
    Convention BiodCase : 2015-02-04T03-00-00_000.wav
    Les '-' dans la partie heure sont remplacés par ':'.
    """
    stem      = wav_path.stem                     # "2015-02-04T03-00-00_000"
    date_part = stem.split("_")[0]                # "2015-02-04T03-00-00"
    date_str, time_str = date_part.split("T")     # "2015-02-04", "03-00-00"
    time_str  = time_str.replace("-", ":")        # "03:00:00"
    return pd.Timestamp(f"{date_str}T{time_str}+00:00")


# ─────────────────────────────────────────────
# SPECTROGRAMME
# ─────────────────────────────────────────────

def compute_spectrogram(wav_path: Path):
    y, sr = librosa.load(wav_path, sr=cfg.SR, mono=True)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.F_MIN,
        fmax=cfg.F_MAX,
    )
    S_db    = librosa.power_to_db(S, ref=np.max, top_db=cfg.DYNAMIC_RANGE)
    hop_sec = cfg.HOP_LENGTH / sr
    return S_db, hop_sec


# ─────────────────────────────────────────────
# EXTRACTION DU PATCH
# ─────────────────────────────────────────────

def extract_patch(
    S_db:        np.ndarray,
    hop_sec:     float,
    audio_start: pd.Timestamp,   # ← passé explicitement, plus de variable globale
    row:         pd.Series,
) -> np.ndarray | None:

    t0 = pd.to_datetime(row["start_datetime"], utc=True)
    t1 = pd.to_datetime(row["end_datetime"],   utc=True)

    # Temps relatifs au début du fichier audio
    t0_rel = (t0 - audio_start).total_seconds()
    t1_rel = (t1 - audio_start).total_seconds()

    # ── Diagnostic silencieux si hors fichier ──────────────────────────
    duration_audio = S_db.shape[1] * hop_sec
    if t1_rel <= 0 or t0_rel >= duration_audio:
        return None   # annotation ne chevauche pas ce fichier
    # ───────────────────────────────────────────────────────────────────

    col0 = int(t0_rel / hop_sec)
    col1 = int(t1_rel / hop_sec)

    freq_axis = librosa.mel_frequencies(n_mels=cfg.N_MELS, fmin=cfg.F_MIN, fmax=cfg.F_MAX or cfg.SR / 2)
    row0 = int(np.argmin(np.abs(freq_axis - row["high_frequency"])))
    row1 = int(np.argmin(np.abs(freq_axis - row["low_frequency"])))

    n_freq, n_frames = S_db.shape
    col0 = max(0, min(col0, n_frames - 1))
    col1 = max(col0 + 1, min(col1, n_frames))
    row0 = max(0, min(row0, n_freq - 1))
    row1 = max(row0 + 1, min(row1, n_freq))

    return S_db[row0:row1, col0:col1]


# ─────────────────────────────────────────────
# NORMALISATION IMAGE
# ─────────────────────────────────────────────

def normalize_patch(patch: np.ndarray) -> np.ndarray:
    h, w = cfg.IMG_SIZE
    p_min, p_max = patch.min(), patch.max()
    if p_max > p_min:
        img = ((patch - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    else:
        img = np.zeros(patch.shape, dtype=np.uint8)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def process_split(split: str):
    audio_root = cfg.DATA_ROOT / split / "audio"
    annot_root = cfg.DATA_ROOT / split / "annotations"
    out_root   = cfg.OUTPUT_ROOT / split

    for annot_file in annot_root.glob("*.csv"):
        df = pd.read_csv(annot_file)
        if df.empty:
            continue

        dataset   = df["dataset"].iloc[0]
        audio_dir = audio_root / dataset

        if not audio_dir.exists():
            logger.warning("Dossier audio introuvable : %s", audio_dir)
            continue

        wav_files = list(audio_dir.glob("*.wav"))
        logger.info("%s → %d wav, %d annotations", dataset, len(wav_files), len(df))

        saved = 0
        for wav in tqdm(wav_files, desc=dataset, leave=False):
            saved += _process_wav(wav, df, out_root)

        logger.info("  → %d imagettes sauvegardées", saved)


def _process_wav(wav_path: Path, df: pd.DataFrame, out_root: Path) -> int:
    filename = wav_path.name
    rows = df[df["filename"].astype(str) == filename]
    if rows.empty:
        return 0

    # Timestamp de début extrait du nom du fichier
    try:
        audio_start = get_audio_start(wav_path)
    except Exception as e:
        logger.warning("Impossible de parser le timestamp de %s : %s", filename, e)
        return 0

    S_db, hop_sec = compute_spectrogram(wav_path)

    count = 0
    for idx, row in rows.iterrows():
        patch = extract_patch(S_db, hop_sec, audio_start, row)
        if patch is None:
            continue

        img   = normalize_patch(patch)
        label = str(row["annotation"]).strip().lower()

        save_dir = out_root / label
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{wav_path.stem}_{idx}.png"), img)
        count += 1

    return count


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("DATA ROOT : %s", cfg.DATA_ROOT)
    logger.info("Existe   : %s", cfg.DATA_ROOT.exists())

    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation"]:
        split_path = cfg.DATA_ROOT / split
        if split_path.exists():
            logger.info("=== Split : %s ===", split)
            process_split(split)
        else:
            logger.warning("Split '%s' introuvable, ignoré.", split)

    logger.info("Terminé → %s", cfg.OUTPUT_ROOT)


if __name__ == "__main__":
    main()