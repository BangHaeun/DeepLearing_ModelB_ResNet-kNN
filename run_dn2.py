"""모델 B 전체 실행.

각 seed마다:
  1) clip 단위 train/val/test split 재생성
  2) 그 seed의 train-normal window만으로 p01/p99 계산
  3) 해당 p01/p99로 ResNet 입력 전처리 및 feature 추출
  4) train-normal feature를 memory bank로 사용
  5) val/test 평가
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from dataset import (
    cond_name,
    load_condition,
    load_config,
    split_clips,
    train_resnet_percentiles,
)
from evaluate import clip_scores, evaluate
from model_dn2 import FeatureExtractor, get_device, knn_scores


def get_features(cfg, extractor, X, meta, name, seed, p01, p99):
    feat_dir = os.path.join(cfg["out_root"], "features")
    os.makedirs(feat_dir, exist_ok=True)

    npz = os.path.join(feat_dir, f"{name}_s{seed}.npz")

    if os.path.exists(npz):
        z = np.load(npz)
        cached_p01 = float(z["train_db_p01"])
        cached_p99 = float(z["train_db_p99"])

        if np.isclose(cached_p01, p01) and np.isclose(cached_p99, p99):
            return (
                z["layer3"],
                z["layer4"],
                float(z["extract_ms_per_clip"]),
            )

    t0 = time.time()
    f = extractor(X, p01, p99, cfg["batch_size"])
    ms = (time.time() - t0) * 1000 / meta["clip_id"].nunique()

    np.savez(
        npz,
        layer3=f["layer3"],
        layer4=f["layer4"],
        extract_ms_per_clip=ms,
        train_db_p01=p01,
        train_db_p99=p99,
        seed=int(seed),
    )

    return f["layer3"], f["layer4"], ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_config(ap.parse_args().config)

    out = cfg["out_root"]
    os.makedirs(os.path.join(out, "scores"), exist_ok=True)

    res_path = os.path.join(out, "dn2_results.csv")
    rows = pd.read_csv(res_path).to_dict("records") if os.path.exists(res_path) else []

    done = {
        (
            r["machine"],
            str(r["id"]).zfill(2),
            int(r["snr"]),
            int(r["sr"]),
            r["layer"],
            int(r["k"]),
            int(r["seed"]),
        )
        for r in rows
        if "seed" in r
    }

    device = get_device()
    print("device:", device)
    extractor = FeatureExtractor(device)
    k = int(cfg["k"])

    for machine, mid, snr, sr in itertools.product(
        cfg["machines"], cfg["ids"], cfg["snrs"], cfg["srs"]
    ):
        name = cond_name(machine, mid, snr, sr)

        try:
            X, base_meta = load_condition(cfg, machine, mid, snr, sr)
        except (FileNotFoundError, ValueError) as e:
            print(f"[skip] {name}: {e}")
            continue

        for seed in cfg["seeds"]:
            try:
                meta = split_clips(base_meta, seed, cfg["val_frac"])
                p01, p99 = train_resnet_percentiles(X, meta)
                f3, f4, ext_ms = get_features(
                    cfg, extractor, X, meta, name, seed, p01, p99
                )
            except ValueError as e:
                print(f"[skip] {name} seed={seed}: {e}")
                continue

            feats = {"layer3": f3, "layer4": f4}

            for layer in cfg["layers"]:
                key = (
                    machine,
                    str(mid).zfill(2),
                    int(snr),
                    int(sr),
                    layer,
                    k,
                    int(seed),
                )
                if key in done:
                    continue

                F_ = feats[layer]
                tr = (meta["split"] == "train").to_numpy()
                ev = ~tr

                bank = F_[tr]

                t0 = time.time()
                s_eval = knn_scores(bank, F_[ev], k=k, device=device)
                n_eval_clips = meta.loc[ev, "clip_id"].nunique()
                knn_ms = (time.time() - t0) * 1000 / n_eval_clips

                eval_meta = meta.loc[ev].reset_index(drop=True)
                clips = clip_scores(eval_meta, s_eval)

                score_path = os.path.join(
                    out, "scores", f"{name}_{layer}_k{k}_s{seed}.csv"
                )
                clips.to_csv(score_path, index=False)

                r = evaluate(clips, cfg["threshold_percentile"])
                r.update(
                    machine=machine,
                    id=str(mid).zfill(2),
                    snr=int(snr),
                    sr=int(sr),
                    layer=layer,
                    k=k,
                    seed=int(seed),
                    train_db_p01=p01,
                    train_db_p99=p99,
                    n_train_windows=int(tr.sum()),
                    extract_ms_per_clip=ext_ms,
                    knn_ms_per_clip=knn_ms,
                )

                rows.append(r)
                pd.DataFrame(rows).to_csv(res_path, index=False)

                print(
                    f"{name} seed={seed} {layer} k={k}: "
                    f"p01={p01:.3f} p99={p99:.3f} "
                    f"AUROC(mean)={r['auroc_mean']:.3f} "
                    f"AUROC(max)={r['auroc_max']:.3f}"
                )

    df = pd.DataFrame(rows)
    if len(df):
        summary = (
            df.groupby(["machine", "id", "snr", "sr", "layer", "k"])[
                ["auroc_mean", "auroc_max"]
            ]
            .agg(["mean", "std"])
            .round(3)
        )
        print(summary)


if __name__ == "__main__":
    main()
