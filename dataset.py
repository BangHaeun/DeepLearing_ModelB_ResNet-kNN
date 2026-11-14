"""전처리된 .npy 윈도우 로드 + 공통 정규화 + 학습/검증/테스트 분할.

공통 정규화는 전처리 담당이 저장한 normalization_stats.json의
training-normal mean/std를 사용한다. 테스트 통계는 다시 계산하지 않는다.
"""
import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

WIN, HOP = 64, 32


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cond_name(machine, mid, snr, sr):
    return f"{machine}_{mid}_{snr}_{sr}"


def clip_files(cfg, machine, mid, snr, sr):
    files = {}
    for label in ("normal", "abnormal"):
        pat = os.path.join(
            cfg["data_root"],
            cfg["layout"].format(snr=snr, machine=machine, id=mid, sr=sr, label=label),
        )
        files[label] = sorted(glob.glob(pat))
    return files


@lru_cache(maxsize=4)
def _read_normalization_stats(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_mean_std(obj, path="root"):
    pairs = []
    if isinstance(obj, dict):
        if "mean" in obj and "std" in obj:
            pairs.append((path, obj["mean"], obj["std"]))
        for key, value in obj.items():
            pairs.extend(_collect_mean_std(value, f"{path}/{key}"))
    return pairs


def _norm_token(x):
    return str(x).lower().replace("+", "").replace("_", "").replace("-", "")


def _resolve_mean_std(stats, machine, mid, snr, sr):
    if isinstance(stats, dict) and "mean" in stats and "std" in stats:
        return stats["mean"], stats["std"]

    pairs = _collect_mean_std(stats)
    if not pairs:
        raise KeyError("normalization_stats.json에서 mean/std를 찾지 못했습니다.")

    machine_t = _norm_token(machine)
    mid_t = _norm_token(mid)
    snr_t = _norm_token(snr)
    sr_t = _norm_token(sr)

    matches = []
    for path, mean, std in pairs:
        p = _norm_token(path)
        if machine_t in p and snr_t in p and sr_t in p:
            matches.append((path, mean, std))

    id_matches = [x for x in matches if mid_t in _norm_token(x[0])]
    if len(id_matches) == 1:
        return id_matches[0][1], id_matches[0][2]
    if len(matches) == 1:
        return matches[0][1], matches[0][2]
    if len(pairs) == 1:
        return pairs[0][1], pairs[0][2]

    available = ", ".join(path for path, _, _ in pairs[:10])
    raise KeyError(
        f"조건({machine}, {mid}, {snr}, {sr})에 맞는 mean/std를 하나로 결정하지 못했습니다. "
        f"stats 경로 예시: {available}"
    )


def _reshape_stat(v, x):
    a = np.asarray(v, dtype=np.float32)
    if a.ndim == 0:
        return a
    if x.ndim == 3 and a.shape == (x.shape[1],):
        return a.reshape(1, -1, 1)
    if x.ndim == 4 and a.shape == (x.shape[2],):
        return a.reshape(1, 1, -1, 1)
    return a


def normalize_windows(x, cfg, machine, mid, snr, sr):
    """training-normal mean/std로 공통 spectrogram 정규화."""
    path = cfg.get("normalization_stats")
    if not path:
        raise KeyError("config.yaml에 normalization_stats 경로가 없습니다.")

    stats = _read_normalization_stats(path)
    mean, std = _resolve_mean_std(stats, machine, mid, snr, sr)

    x = np.asarray(x, dtype=np.float32)
    mean = _reshape_stat(mean, x)
    std = _reshape_stat(std, x)

    if np.any(np.asarray(std) <= 0):
        raise ValueError("normalization std는 0보다 커야 합니다.")

    return (x - mean) / std


def to_windows(a):
    """파일 하나 -> (n_win, 128, 64). 이미 윈도우면 그대로, 전체 spectrogram이면 자른다."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4:                       # 기존 코드의 (n,1,128,64) 형식
        if a.shape[1] != 1:
            raise ValueError(
                f"4차원 입력 shape={a.shape}. 현재 실제 데이터의 (clip,8,128,64) 구조는 "
                "manifest 기반 로더로 별도 처리해야 합니다."
            )
        a = a[:, 0]
    if a.ndim == 2:                       # (128, T)
        starts = range(0, a.shape[1] - WIN + 1, HOP)
        a = np.stack([a[:, s:s + WIN] for s in starts])
    if a.ndim != 3:
        raise ValueError(f"예상 밖의 shape: {a.shape}")
    return a


def load_condition(cfg, machine, mid, snr, sr):
    """한 조건의 모든 클립 -> X (N,128,64), meta(clip_id, label, window_idx)."""
    files = clip_files(cfg, machine, mid, snr, sr)
    if not files["normal"] or not files["abnormal"]:
        raise FileNotFoundError(
            f"파일을 못 찾음: normal={len(files['normal'])}, abnormal={len(files['abnormal'])}. "
            f"config.yaml의 data_root/layout을 확인하세요."
        )

    xs, rows = [], []
    for label_name, label in (("normal", 0), ("abnormal", 1)):
        for p in files[label_name]:
            raw = np.load(p)
            raw = normalize_windows(raw, cfg, machine, mid, snr, sr)
            w = to_windows(raw)

            clip_id = f"{label_name}/{os.path.basename(p)}"
            xs.append(w)
            rows += [(clip_id, label, i) for i in range(len(w))]

    X = np.concatenate(xs)
    meta = pd.DataFrame(rows, columns=["clip_id", "label", "window_idx"])
    return X, meta


def split_clips(meta, seed, val_frac=0.1):
    """meta에 'split' 열(train/val/test)을 붙여 반환."""
    rng = np.random.default_rng(seed)
    clips = meta.drop_duplicates("clip_id")
    ab = clips[clips.label == 1].clip_id.tolist()
    no = clips[clips.label == 0].clip_id.tolist()
    rng.shuffle(no)
    n_test = min(len(ab), len(no) // 2)
    test_ids = set(ab) | set(no[:n_test])
    rest = no[n_test:]
    n_val = max(1, int(round(len(rest) * val_frac)))
    val_ids = set(rest[:n_val])

    def which(c):
        return "test" if c in test_ids else ("val" if c in val_ids else "train")

    out = meta.copy()
    out["split"] = out.clip_id.map(which)
    return out
