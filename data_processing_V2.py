# filepath: data_processing.py

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class Config:
    BASE_DIR    = Path(__file__).resolve().parent
    DATA_ROOT   = Path(r"C:\ENSTA\2A\S4\Machine_learning\Projet\biodcase_development_set\biodcase_development_set")
    OUTPUT_ROOT = BASE_DIR / "dataset_prepared"

    SR            = 250
    N_FFT         = 512      # 0.49 Hz/bin → même résolution que pipeline de référence
    HOP_LENGTH    = 64       # frame = 256 ms
    F_MIN         = 0
    F_MAX         = None

    IMG_SIZE      = (128, 128)
    DYNAMIC_RANGE = 80



cfg = Config()


def get_audio_start(wav_path: Path) -> pd.Timestamp:
    stem      = wav_path.stem
    date_part = stem.split("_")[0]
    date_str, time_str = date_part.split("T")
    time_str  = time_str.replace("-", ":")
    return pd.Timestamp(f"{date_str}T{time_str}+00:00")


def compute_spectrogram(wav_path: Path):
    sr_orig, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr_orig != cfg.SR:
        num_samples = int(len(data) * cfg.SR / sr_orig)
        data = np.interp(
            np.linspace(0, len(data), num_samples),
            np.arange(len(data)),
            data,
        ).astype(np.float32)
    sr = cfg.SR

    freqs, times, Sxx = scipy_spectrogram(
        data,
        fs       = sr,
        window   = "hann",
        nperseg  = cfg.N_FFT,
        noverlap = cfg.N_FFT - cfg.HOP_LENGTH,
        scaling  = "density",
        mode     = "psd",
    )

    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-10))
    Sxx_db = Sxx_db - Sxx_db.max()
    Sxx_db = np.maximum(Sxx_db, -cfg.DYNAMIC_RANGE)

    hop_sec = cfg.HOP_LENGTH / sr
    return Sxx_db, hop_sec, freqs


def extract_patch(
    S_db:        np.ndarray,
    hop_sec:     float,
    audio_start: pd.Timestamp,
    freqs:       np.ndarray,
    row:         pd.Series,
) -> np.ndarray | None:
    """
    Crop exact sur la bbox d'annotation — même logique que le pipeline de référence.
    Utilise np.searchsorted comme lui pour trouver les indices temps et fréquence.
    """
    t0 = pd.to_datetime(row["start_datetime"], utc=True)
    t1 = pd.to_datetime(row["end_datetime"],   utc=True)

    # Temps relatifs au début du fichier audio
    t_start = (t0 - audio_start).total_seconds()
    t_end   = (t1 - audio_start).total_seconds()

    # Axe temporel en secondes (même que librosa.frames_to_time)
    n_frames = S_db.shape[1]
    times    = np.arange(n_frames) * hop_sec

    # Indices temps — searchsorted comme dans le code de référence
    col_start = np.searchsorted(times, t_start)
    col_end   = np.searchsorted(times, t_end)

    # Indices fréquence — fréquences croissantes, low → bas, high → haut
    row_low  = np.searchsorted(freqs, row["low_frequency"])
    row_high = np.searchsorted(freqs, row["high_frequency"])

    # Vérification des bornes
    if col_start >= col_end or row_low >= row_high:
        return None
    if col_end > S_db.shape[1] or row_high > S_db.shape[0]:
        return None

    # Crop exact sur la bbox
    patch = S_db[row_low:row_high, col_start:col_end]

    # Inverser l'axe fréquence (basses fréquences en bas) — comme lui
    patch = patch[::-1, :]

    return patch


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalisation simple min/max → [0, 255], niveaux de gris.
    Même approche que le pipeline librosa de référence (camarade).
    Pas de soustraction de médiane qui détruit le signal sur les petits patches.
    """
    h, w = cfg.IMG_SIZE

    p_min, p_max = patch.min(), patch.max()
    if p_max - p_min < 1e-6:
        img = np.zeros(patch.shape, dtype=np.uint8)
    else:
        img = ((patch - p_min) / (p_max - p_min) * 255).astype(np.uint8)

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


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

    try:
        audio_start = get_audio_start(wav_path)
    except Exception as e:
        logger.warning("Impossible de parser le timestamp de %s : %s", filename, e)
        return 0

    S_db, hop_sec, freqs = compute_spectrogram(wav_path)

    count = 0
    for idx, row in rows.iterrows():
        patch = extract_patch(S_db, hop_sec, audio_start, freqs, row)
        if patch is None:
            continue

        img   = normalize_patch(patch)
        label = str(row["annotation"]).strip().lower()

        save_dir = out_root / label
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{wav_path.stem}_{idx}.png"), img)
        count += 1

    return count


def main():
    logger.info("DATA ROOT : %s", cfg.DATA_ROOT)
    logger.info("Existe    : %s", cfg.DATA_ROOT.exists())

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