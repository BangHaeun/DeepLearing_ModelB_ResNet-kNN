"""전처리된 MIMII feature 묶음 + manifest 로더.

실제 배포 형식:
  project/features/{machine}_{snr}dB_{sr}.npy  -> (n_clips, 8, 128, 64)
  project/manifest.csv                        -> clip 메타데이터 + feature_index
  project/normalization_stats.json            -> train-normal 기반 모델별 정규화 통계

manifest의 split(train/val/test)을 그대로 사용한다.
"""

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _snr_tag(snr):
    snr = int(snr)
    return f"{snr:+d}dB"


def cond_name(machine, mid, snr, sr):
    return f"{machine}_{str(mid).zfill(2)}_{_snr_tag(snr)}_{int(sr)}"


@lru_cache(maxsize=2)
def _load_manifest(path):
    return pd.read_csv(path)


@lru_cache(maxsize=2)
def _load_stats(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _group_key(machine, mid, snr, sr):
    return f"{machine}/{str(mid).zfill(2)}/{_snr_tag(snr)}/{int(sr)}"


def get_resnet_scaling(cfg, machine, mid, snr, sr):
    """해당 조건의 train-normal p01/p99를 반환."""
    stats = _load_stats(cfg["normalization_stats"])
    key = _group_key(machine, mid, snr, sr)
    try:
        g = stats["groups"][key]
    except KeyError as e:
        raise KeyError(f"normalization_stats.json에 조건 {key}가 없습니다.") from e

    return float(g["train_db_p01"]), float(g["train_db_p99"])


def load_condition(cfg, machine, mid, snr, sr):
    """한 조건의 모든 clip을 window 단위로 펼쳐 반환.

    Returns
    -------
    X : (n_clips * 8, 128, 64) float32
    meta : 각 window의 clip_id, label, split, window_idx
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
        raise ValueError(f"조건 하나에 feature_file이 {len(feature_files)}개입니다: {feature_files}")

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
    splits = sub["split"].astype(str).to_numpy()
    clip_ids = sub["clip_key"].astype(str).to_numpy()

    meta = pd.DataFrame({
        "clip_id": np.repeat(clip_ids, n_win),
        "label": np.repeat(labels, n_win),
        "split": np.repeat(splits, n_win),
        "window_idx": np.tile(np.arange(n_win), n_clips),
    })

    return X, meta
