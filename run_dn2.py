"""모델 B 전체 실행: 조건(기계×ID×SNR×sr) × 층 × 시드.

산출물 (out_root 아래):
  features/{cond}.npz, features/{cond}_meta.csv   특징 캐시 (조건당 1회만 추출)
  scores/{cond}_{layer}_k{k}_s{seed}.csv          클립 점수 (clip_id,label,split,score_mean,score_max)
  dn2_results.csv                                 요약 (AUROC 등)  <- 분석 담당이 읽는 파일
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from dataset import cond_name, load_condition, load_config, split_clips
from evaluate import clip_scores, evaluate
from model_dn2 import FeatureExtractor, get_device, knn_scores


def get_features(cfg, extractor, machine, mid, snr, sr):
    feat_dir = os.path.join(cfg["out_root"], "features")
    os.makedirs(feat_dir, exist_ok=True)
    name = cond_name(machine, mid, snr, sr)
    npz, meta_csv = os.path.join(feat_dir, f"{name}.npz"), os.path.join(feat_dir, f"{name}_meta.csv")
    if os.path.exists(npz) and os.path.exists(meta_csv):
        z = np.load(npz)
        return z["layer3"], z["layer4"], pd.read_csv(meta_csv), float(z["extract_ms_per_clip"])
    X, meta = load_condition(cfg, machine, mid, snr, sr)
    t0 = time.time()
    f = extractor(X, cfg["batch_size"])
    ms = (time.time() - t0) * 1000 / meta.clip_id.nunique()
    np.savez(npz, layer3=f["layer3"], layer4=f["layer4"], extract_ms_per_clip=ms)
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
    done = {(r["machine"], str(r["id"]).zfill(2), r["snr"], int(r["sr"]), r["layer"], int(r["k"]), int(r["seed"]))
            for r in rows}

    device = get_device()
    print("device:", device)
    extractor = FeatureExtractor(device)
    k = cfg["k"]

    for machine, mid, snr, sr in itertools.product(cfg["machines"], cfg["ids"], cfg["snrs"], cfg["srs"]):
        name = cond_name(machine, mid, snr, sr)
        try:
            f3, f4, meta, ext_ms = get_features(cfg, extractor, machine, mid, snr, sr)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        feats = {"layer3": f3, "layer4": f4}

        for layer, seed in itertools.product(cfg["layers"], cfg["seeds"]):
            key = (machine, mid, snr, int(sr), layer, int(k), int(seed))
            if key in done:
                continue
            m = split_clips(meta, seed, cfg["val_frac"])
            F_ = feats[layer]
            tr = (m.split == "train").values
            bank = F_[tr]                                        # 메모리 뱅크: 학습 정상 윈도우 전부
            t0 = time.time()
            s_all = np.zeros(len(m), dtype=np.float32)
            ev = (~tr)
            s_all[ev] = knn_scores(bank, F_[ev], k=k, device=device)
            n_eval_clips = m[ev].clip_id.nunique()
            knn_ms = (time.time() - t0) * 1000 / n_eval_clips

            clips = clip_scores(m[ev].reset_index(drop=True), s_all[ev])
            clips.to_csv(os.path.join(out, "scores", f"{name}_{layer}_k{k}_s{seed}.csv"), index=False)
            r = evaluate(clips, cfg["threshold_percentile"])
            r.update(machine=machine, id=mid, snr=snr, sr=sr, layer=layer, k=k, seed=seed,
                     n_train_windows=int(tr.sum()), extract_ms_per_clip=ext_ms, knn_ms_per_clip=knn_ms)
            rows.append(r)
            pd.DataFrame(rows).to_csv(res_path, index=False)    # 매번 저장 (Colab 끊김 대비)
            print(f"{name} {layer} k={k} seed={seed}: AUROC(mean)={r['auroc_mean']:.3f}  AUROC(max)={r['auroc_max']:.3f}")

    df = pd.DataFrame(rows)
    if len(df):
        print(df.groupby(["machine", "id", "snr", "sr", "layer", "k"])[["auroc_mean", "auroc_max"]]
              .agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
