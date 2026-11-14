# MIMII Model B: ResNet18 + kNN

## 데이터 구조

전처리 bundle은 조건별 clip을 하나의 배열에 묶어 저장한다.

```text
project/features/fan_+6dB_16000.npy
shape = (n_clips, 8, 128, 64)
```

`project/manifest.csv`의 `feature_file`, `feature_index`를 사용해 각 clip을 feature 배열과 연결한다.

## Seed별 split

manifest에 들어 있는 기존 `split` 값은 실험에 사용하지 않는다.

각 seed마다 clip 단위로 다시 나눈다.

```text
abnormal 전부 -> test
normal 중 abnormal과 같은 개수 -> test
남은 normal -> train 90% / val 10%
```

seed가 바뀌면 normal clip의 test/val/train 구성이 달라진다.

## Model B 입력 정규화

각 seed에서 새로 정해진 train-normal window만 사용해 p01/p99를 다시 계산한다.

```text
log-mel dB window
-> seed별 train-normal p01/p99
-> p01~p99를 [0,1]로 scaling + clipping
-> 3채널 복사
-> 224x224 bilinear resize
-> ImageNet mean/std normalization
-> pretrained ResNet18
```

따라서 test/val 데이터는 p01/p99 계산에 사용되지 않는다.

p01/p99가 seed마다 달라지므로 ResNet feature도 seed마다 다시 추출하며,
feature cache 파일명에도 seed를 포함한다.

```text
features/{condition}_s0.npz
features/{condition}_s1.npz
features/{condition}_s2.npz
```

## Colab 실행

```python
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/MIMII_project_preprocessed.zip" /content/
!mkdir -p /content/MIMII_project_preprocessed
!unzip -q /content/MIMII_project_preprocessed.zip -d /content/MIMII_project_preprocessed

!git clone https://github.com/BangHaeun/DeepLearing_ModelB_ResNet-kNN.git
%cd DeepLearing_ModelB_ResNet-kNN

!pip install -r requirements.txt
!python run_dn2.py --config config.yaml
```

최종 `dn2_results.csv`에는 seed별 결과가 저장되고,
출력 요약은 3개 seed의 AUROC mean ± std를 계산한다.
