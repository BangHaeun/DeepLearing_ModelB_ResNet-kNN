# MIMII Model B: ResNet18 + kNN

## 실제 전처리 데이터 구조

전처리 bundle은 조건별 clip을 하나의 배열에 묶어 저장한다.

```text
project/features/fan_+6dB_16000.npy
shape = (n_clips, 8, 128, 64)
```

`project/manifest.csv`의 `feature_file`, `feature_index`가 각 clip과 feature 배열 row를 연결한다.
manifest에는 이미 `train / val / test` split도 저장되어 있으므로 이 split을 그대로 사용한다.

## Model B 입력 정규화

`normalization_stats.json`에 따르면 Model B(ResNet18)는 ConvAE와 다른 입력 정규화를 사용한다.

```text
log-mel dB window
-> 해당 machine / device ID / SNR / sample rate의
   train-normal p01, p99 사용
-> p01~p99를 [0,1]로 scaling + clipping
-> 3채널 복사
-> 224x224 bilinear resize
-> ImageNet mean/std normalization
-> pretrained ResNet18
```

ConvAE용 per-mel mean/std는 Model B에서 사용하지 않는다.

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

결과는 `out_root`에 저장된다.

## 현재 split 관련 주의

전처리 bundle의 normalization 통계는 manifest에 저장된 train split 기준으로 계산되어 있다.
따라서 현재 코드는 manifest split을 그대로 사용한다.

계획서의 3 random seeds 실험을 하려면 seed마다 train/val/test를 다시 나누고,
각 seed의 train-normal에서 p01/p99도 다시 계산해야 한다.
