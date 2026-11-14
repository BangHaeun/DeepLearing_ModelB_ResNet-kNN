"""모델 B 전체 실행.

manifest에 저장된 train/val/test split을 그대로 사용한다.
전처리 bundle의 normalization_stats도 이 split의 train-normal로 계산되어 있다.
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from dataset import cond_name, get_resnet_scaling, load_condition, load_config
from evaluate import clip_scores, evaluate
from model_dn2 import FeatureExtractor, get_device, knn_scores


def get_features(cfg, extractor, machine, mid, snr, sr):
    feat_dir = os.path.join(cfg["out_root"], "features")
    os.makedirs(feat_dir, exist_ok=True)

    name = cond_name(machine, mid, snr, sr)
    npz = os.path.join(feat_dir, f"{name}.npz")
    meta_csv = os.path.join(feat_dir, f"{name}_meta.csv")

    if os.path.exists(npz) and os.path.exists(meta_csv):
        z = np.load(npz)
        return (
            z["layer3"],
            z["layer4"],
            pd.read_csv(meta_csv),
            float(z["extract_ms_per_clip"]),
        )

    X, meta = load_condition(cfg, machine, mid, snr, sr)
    p01, p99 = get_resnet_scaling(cfg, machine, mid, snr, sr)

    t0 = time.time()
    f = extractor(X, p01, p99, cfg["batch_size"])
    ms = (time.time() - t0) * 1000 / meta.clip_id.nunique()

    np.savez(
        npz,
        layer3=f["layer3"],
        layer4=f["layer4"],
        extract_ms_per_clip=ms,
        train_db_p01=p01,
        train_db_p99=p99,
    )
    meta.to_csv(meta_csv, index=False)

    return f["layer3"], f["layer4"], meta, ms


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
        )
        for r in rows
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
            f3, f4, meta, ext_ms = get_features(
                cfg, extractor, machine, mid, snr, sr
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[skip] {name}: {e}")
            continue

        feats = {"layer3": f3, "layer4": f4}

        for layer in cfg["layers"]:
            key = (machine, str(mid).zfill(2), int(snr), int(sr), layer, k)
            if key in done:
                continue

            F_ = feats[layer]
            tr = (meta["split"] == "train").to_numpy()
            ev = ~tr

            # manifest의 train은 정상만이어야 한다.
            if np.any(meta.loc[tr, "label"].to_numpy() != 0):
                raise ValueError(f"{name}: train split에 abnormal이 포함되어 있습니다.")

            bank = F_[tr]

            t0 = time.time()
            s_eval = knn_scores(bank, F_[ev], k=k, device=device)
            n_eval_clips = meta.loc[ev, "clip_id"].nunique()
            knn_ms = (time.time() - t0) * 1000 / n_eval_clips

            eval_meta = meta.loc[ev].reset_index(drop=True)
            clips = clip_scores(eval_meta, s_eval)

            score_path = os.path.join(
                out, "scores", f"{name}_{layer}_k{k}.csv"
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
                n_train_windows=int(tr.sum()),
                extract_ms_per_clip=ext_ms,
                knn_ms_per_clip=knn_ms,
            )

            rows.append(r)
            pd.DataFrame(rows).to_csv(res_path, index=False)

            print(
                f"{name} {layer} k={k}: "
                f"AUROC(mean)={r['auroc_mean']:.3f} "
                f"AUROC(max)={r['auroc_max']:.3f}"
            )

    df = pd.DataFrame(rows)
    if len(df):
        print(
            df.groupby(["machine", "id", "snr", "sr", "layer", "k"])[
                ["auroc_mean", "auroc_max"]
            ]
            .mean()
            .round(3)
        )


if __name__ == "__main__":
    main()
