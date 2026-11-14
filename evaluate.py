"""윈도우 점수 -> 클립 점수 -> AUROC / 임계치."""
import numpy as np
from sklearn.metrics import roc_auc_score


def clip_scores(meta, win_scores):
    """meta: clip_id,label,split 열 포함. 윈도우 점수를 클립 단위로 평균·최댓값 집계."""
    df = meta.assign(score=win_scores)
    g = df.groupby("clip_id").agg(
        label=("label", "first"),
        split=("split", "first"),
        score_mean=("score", "mean"),
        score_max=("score", "max"),
    )
    return g.reset_index()


def evaluate(clips, pct=95):
    test = clips[clips.split == "test"]
    val = clips[clips.split == "val"]
    thr = float(np.percentile(val.score_mean, pct))     # 임계치는 검증 정상으로만 결정
    pred = test.score_mean > thr
    y = test.label.values
    return {
        "auroc_mean": roc_auc_score(y, test.score_mean),
        "auroc_max": roc_auc_score(y, test.score_max),
        "threshold": thr,
        "tpr": float(pred[y == 1].mean()),
        "fpr": float(pred[y == 0].mean()),
        "n_test_normal": int((y == 0).sum()),
        "n_test_abnormal": int((y == 1).sum()),
    }
