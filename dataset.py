"""전처리된 MIMII feature 묶음 + manifest 로더.

실제 배포 형식:
  project/features/{machine}_{snr}dB_{sr}.npy  -> (n_clips, 8, 128, 64)
  project/manifest.csv                        -> clip 메타데이터 + feature_index

manifest의 기존 split 열은 사용하지 않는다.
각 random seed마다 clip 단위 split을 새로 만든다.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _snr_tag(snr):
    return f"{int(snr):+d}dB"


def cond_name(machine, mid, snr, sr):
    return f"{machine}_{str(mid).zfill(2)}_{_snr_tag(snr)}_{int(sr)}"


@lru_cache(maxsize=2)
def _load_manifest(path):
    return pd.read_csv(path)


def load_condition(cfg, machine, mid, snr, sr):
    """한 조건의 모든 clip을 window 단위로 펼쳐 반환.

    Returns
    -------
    X : (n_clips * 8, 128, 64) float32
    meta : 각 window의 clip_id, label, window_idx
    """
    manifest = _load_manifest(cfg["manifest"])

    sub = manifest[
        (manifest["machine"] == machine)
        & (manifest["device_id"] == int(mid))
        & (manifest["snr_db"] == int(snr))
        & (manifest["sample_rate"] == int(sr))
    ].copy()

    if sub.empty:
        raise FileNotFoundError(
            f"manifest에서 조건을 찾지 못함: machine={machine}, id={mid}, snr={snr}, sr={sr}"
        )

    feature_files = sub["feature_file"].dropna().unique()
    if len(feature_files) != 1:
        raise ValueError(
            f"조건 하나에 feature_file이 {len(feature_files)}개입니다: {feature_files}"
        )

    feature_path = os.path.join(cfg["project_root"], feature_files[0])
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"feature 파일을 찾지 못함: {feature_path}")

    all_clips = np.load(feature_path, mmap_mode="r")
    idx = sub["feature_index"].astype(int).to_numpy()
    clips = np.asarray(all_clips[idx], dtype=np.float32)

    if clips.ndim != 4 or clips.shape[1:] != (8, 128, 64):
        raise ValueError(f"예상 shape (n_clips,8,128,64), 실제 {clips.shape}")

    n_clips, n_win, _, _ = clips.shape
    X = clips.reshape(n_clips * n_win, 128, 64)

    labels = (sub["label"].str.lower() == "abnormal").astype(np.int64).to_numpy()
    clip_ids = sub["clip_key"].astype(str).to_numpy()

    meta = pd.DataFrame({
        "clip_id": np.repeat(clip_ids, n_win),
        "label": np.repeat(labels, n_win),
        "window_idx": np.tile(np.arange(n_win), n_clips),
    })

    return X, meta


def split_clips(meta, seed, val_frac=0.1):
    """seed마다 clip 단위 train/val/test split을 새로 만든다.

    계획서 규칙:
      test = abnormal 전부 + 같은 개수의 normal
      나머지 normal = train / val
    """
    rng = np.random.default_rng(int(seed))
    clips = meta.drop_duplicates("clip_id")[["clip_id", "label"]]

    abnormal = clips.loc[clips["label"] == 1, "clip_id"].to_numpy()
    normal = clips.loc[clips["label"] == 0, "clip_id"].to_numpy().copy()

    rng.shuffle(normal)

    n_test_normal = len(abnormal)
    if len(normal) <= n_test_normal:
        raise ValueError(
            f"정상 clip({len(normal)})이 고장 clip({len(abnormal)})보다 충분하지 않아 "
            "test normal을 같은 개수로 뽑고 train/val을 남길 수 없습니다."
        )

    test_normal = set(normal[:n_test_normal])
    remain_normal = normal[n_test_normal:]

    n_val = max(1, int(round(len(remain_normal) * float(val_frac))))
    if n_val >= len(remain_normal):
        raise ValueError("val 분할 후 train normal clip이 남지 않습니다.")

    val_ids = set(remain_normal[:n_val])
    test_ids = set(abnormal) | test_normal

    out = meta.copy()

    def which(clip_id):
        if clip_id in test_ids:
            return "test"
        if clip_id in val_ids:
            return "val"
        return "train"

    out["split"] = out["clip_id"].map(which)

    if np.any(out.loc[out["split"] == "train", "label"].to_numpy() != 0):
        raise ValueError("train split에 abnormal clip이 포함되었습니다.")

    return out


def train_resnet_percentiles(X, meta):
    """현재 seed의 train-normal window만으로 p01/p99 계산."""
    train_mask = (meta["split"] == "train").to_numpy()
    if not np.any(train_mask):
        raise ValueError("train window가 없습니다.")

    train_x = X[train_mask]
    p01, p99 = np.percentile(train_x, [1, 99])

    if not np.isfinite(p01) or not np.isfinite(p99) or p99 <= p01:
        raise ValueError(f"잘못된 p01/p99: p01={p01}, p99={p99}")

    return float(p01), float(p99)
