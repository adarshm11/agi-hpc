"""BirdCLEF 2026 — Train classifier on geometric features (SPD + TDA).

Pipeline:
1. Extract 156-dim geometric features from all training audio (CPU, parallel)
2. Train a LightGBM/sklearn classifier on the features
3. Evaluate with AUC on validation set
4. Log results

Runs entirely on CPU — does not interfere with Nemotron GPU training.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

DATA_DIR = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/birdclef/data")))
FEATURE_DIR = Path(os.environ.get("FEATURE_DIR", os.path.expanduser("~/birdclef/precomputed/geometric")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", os.path.expanduser("~/birdclef/output/geometric")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "geometric_train.log"

N_WORKERS = int(os.environ.get("N_WORKERS", 0)) or max(1, cpu_count() - 2)  # leave 2 CPUs free
N_FOLDS = 5
SEED = 42

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ═══════════════════════════════════════════════════════════════
# Step 1: Extract geometric features
# ═══════════════════════════════════════════════════════════════

def process_one_file(args_tuple):
    """Extract geometric features from one audio file."""
    audio_path, output_path = args_tuple
    try:
        import torch
        import torchaudio
        from src.data.audio import load_audio, extract_window
        from src.data.geometric_features import extract_geometric_features

        waveform = load_audio(audio_path)
        window = extract_window(waveform, offset=0)

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
        db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        spec = db(mel(window.unsqueeze(0)))
        spec_np = spec.numpy()
        if spec_np.ndim == 3:
            spec_np = spec_np[0]  # remove batch dim: (1, 128, T) -> (128, T)

        features = extract_geometric_features(
            waveform=window.numpy(),
            spectrogram=spec_np,
            n_bands=16,
            tda_delay=10,
            tda_dim=3,
            tda_max_points=500,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        return True
    except Exception as e:
        return False


def extract_all_features():
    """Extract geometric features for all training files."""
    log(f"=== STEP 1: Feature Extraction (CPUs: {N_WORKERS}) ===")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    tasks = []
    already_done = 0

    for _, row in train_df.iterrows():
        filename = row["filename"]
        audio_path = str(DATA_DIR / "train_audio" / filename)
        stem = Path(filename).stem
        label = str(row["primary_label"])
        out_path = str(FEATURE_DIR / label / f"{stem}.npy")

        if os.path.exists(out_path):
            already_done += 1
        else:
            tasks.append((audio_path, out_path))

    log(f"  Total files: {len(train_df)}")
    log(f"  Already extracted: {already_done}")
    log(f"  Remaining: {len(tasks)}")

    if not tasks:
        log("  All features already computed!")
        return

    t0 = time.time()
    success = 0
    fail = 0

    with Pool(N_WORKERS) as pool:
        for i, ok in enumerate(pool.imap_unordered(process_one_file, tasks)):
            if ok:
                success += 1
            else:
                fail += 1
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(tasks) - i - 1) / rate / 60
                log(f"  [{i+1}/{len(tasks)}] {success} ok, {fail} fail, "
                    f"{rate:.1f} files/sec, ~{remaining:.0f} min remaining")

    elapsed = time.time() - t0
    log(f"  Extraction complete: {elapsed/60:.1f} min, {success} ok, {fail} fail")


# ═══════════════════════════════════════════════════════════════
# Step 2: Load features into arrays
# ═══════════════════════════════════════════════════════════════

def load_feature_matrix():
    """Load all precomputed features into X, y arrays."""
    log("=== STEP 2: Loading feature matrix ===")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    le = LabelEncoder()
    all_labels = le.fit_transform(train_df["primary_label"])

    X_list = []
    y_list = []
    skipped = 0

    for idx, row in train_df.iterrows():
        filename = row["filename"]
        stem = Path(filename).stem
        label = str(row["primary_label"])
        feat_path = FEATURE_DIR / label / f"{stem}.npy"

        if feat_path.exists():
            feat = np.load(feat_path)
            if feat.shape[0] > 0 and np.isfinite(feat).all():
                X_list.append(feat)
                y_list.append(all_labels[idx])
            else:
                skipped += 1
        else:
            skipped += 1

    X = np.stack(X_list)
    y = np.array(y_list)

    log(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    log(f"  Skipped: {skipped}")

    return X, y, le


# ═══════════════════════════════════════════════════════════════
# Step 3: Train classifier
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate(X, y, le):
    """Train LightGBM with stratified k-fold, report AUC."""
    log(f"=== STEP 3: Training ({N_FOLDS}-fold CV) ===")

    n_classes = len(le.classes_)

    # Try LightGBM first, fall back to sklearn
    try:
        import lightgbm as lgb
        USE_LGB = True
        log("  Using LightGBM")
    except ImportError:
        USE_LGB = False
        log("  LightGBM not available, using sklearn GradientBoosting")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        t0 = time.time()

        if USE_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=500,
                num_leaves=63,
                learning_rate=0.05,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=N_WORKERS,
                random_state=SEED + fold,
                verbose=-1,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=SEED + fold,
            )
            model.fit(X_train, y_train)

        # Predict probabilities
        y_pred = model.predict_proba(X_val)

        # Compute macro AUC
        try:
            auc = roc_auc_score(y_val, y_pred, multi_class="ovr", average="macro")
        except ValueError:
            # Some classes may not appear in val fold
            auc = roc_auc_score(y_val, y_pred, multi_class="ovr", average="weighted")

        elapsed = time.time() - t0
        fold_aucs.append(auc)
        log(f"  Fold {fold+1}/{N_FOLDS}: AUC={auc:.4f} ({elapsed:.0f}s)")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    log(f"  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    return mean_auc, std_auc, fold_aucs


# ═══════════════════════════════════════════════════════════════
# Step 4: Report results
# ═══════════════════════════════════════════════════════════════

def save_results(mean_auc, std_auc, fold_aucs):
    """Save results and write completion marker."""
    results = {
        "model": "geometric_lgbm",
        "features": "SPD_manifold_156d",
        "feature_breakdown": {
            "spd_covariance": 136,
            "spectral_trajectory": 4,
            "tda_persistent_homology": 16,
        },
        "n_folds": N_FOLDS,
        "mean_auc": float(mean_auc),
        "std_auc": float(std_auc),
        "fold_aucs": [float(a) for a in fold_aucs],
        "baseline_auc": 0.5001,
        "improvement": f"{(mean_auc - 0.5001) * 100:.1f} percentage points",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = OUTPUT_DIR / "geometric_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"=== RESULTS ===")
    log(f"  Geometric features AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    log(f"  Baseline (raw spectrogram): 0.5001")
    log(f"  Improvement: {(mean_auc - 0.5001) * 100:.1f} pp")
    log(f"  Results saved to: {results_path}")
    log(f"=== DONE ===")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log("=" * 60)
    log("BirdCLEF 2026 — Geometric Feature Pipeline")
    log("SPD manifold (136d) + spectral trajectory (4d) + TDA (16d) = 156d")
    log("=" * 60)

    # Step 1: Extract features (CPU parallel, no GPU)
    extract_all_features()

    # Step 2: Load into arrays
    X, y, le = load_feature_matrix()

    # Step 3: Train and evaluate
    mean_auc, std_auc, fold_aucs = train_and_evaluate(X, y, le)

    # Step 4: Save results
    save_results(mean_auc, std_auc, fold_aucs)
