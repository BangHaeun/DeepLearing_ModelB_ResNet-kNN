"""전처리된 .npy 윈도우 로드 + 학습/검증/테스트 분할 (계획서 규칙).

분할: 테스트 = 고장 전부 + 같은 수의 정상(랜덤), 나머지 정상 = 학습 90% / 검증 10%.
분할은 '클립' 단위로 한다 (같은 클립의 윈도우가 학습·테스트에 섞이지 않도록).
"""
import glob
import os

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


def to_windows(a):
    """파일 하나 -> (n_win, 128, 64). 이미 윈도우면 그대로, 전체 spectrogram(128, T)이면 자른다."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4:                       # (n,1,128,64)
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
            w = to_windows(np.load(p))
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
    n_test = min(len(ab), len(no) // 2)   # 정상이 너무 적을 때 학습용을 남기기 위한 안전장치
    test_ids = set(ab) | set(no[:n_test])
    rest = no[n_test:]
    n_val = max(1, int(round(len(rest) * val_frac)))
    val_ids = set(rest[:n_val])

    def which(c):
        return "test" if c in test_ids else ("val" if c in val_ids else "train")

    out = meta.copy()
    out["split"] = out.clip_id.map(which)
    return out
