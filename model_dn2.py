"""모델 B: ImageNet 사전학습 ResNet18 특징 + kNN 거리."""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """윈도우 (N,128,64) -> layer3 256D / layer4 512D."""

    def __init__(self, device=None):
        self.device = device or get_device()
        self.m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(self.device)
        for p in self.m.parameters():
            p.requires_grad = False
        self.mean = IMAGENET_MEAN.to(self.device)
        self.std = IMAGENET_STD.to(self.device)

    def _prep(self, xb, p01, p99):
        # 전처리 담당 규칙:
        # train-normal의 p01/p99로 clip -> [0,1] -> 3채널 -> resize -> ImageNet normalize
        denom = max(float(p99) - float(p01), 1e-6)
        x = ((xb - float(p01)) / denom).clamp_(0.0, 1.0)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, X, p01, p99, batch_size=256):
        f3, f4 = [], []
        m = self.m

        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().to(self.device)
            x = self._prep(xb, p01, p99)

            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            x = m.layer2(m.layer1(x))
            x3 = m.layer3(x)
            x4 = m.layer4(x3)

            f3.append(x3.mean(dim=(2, 3)).cpu())
            f4.append(x4.mean(dim=(2, 3)).cpu())

        return {
            "layer3": torch.cat(f3).numpy(),
            "layer4": torch.cat(f4).numpy(),
        }


@torch.no_grad()
def knn_scores(bank, query, k=1, chunk=1024, device=None):
    """각 query의 k개 최근접 정상 feature까지 평균 Euclidean 거리."""
    device = device or get_device()
    b = torch.from_numpy(bank).float().to(device)
    out = []

    for i in range(0, len(query), chunk):
        q = torch.from_numpy(query[i:i + chunk]).float().to(device)
        d = torch.cdist(q, b)
        out.append(d.topk(k, dim=1, largest=False).values.mean(dim=1).cpu())

    return torch.cat(out).numpy()
