# MIMII DN2 (ResNet18 + kNN) 이상탐지

## 정규화 규칙

전처리된 log-mel 값 자체는 원본 dB 범위로 저장되어 있다.
모델 B는 전처리 담당이 저장한 `project/normalization_stats.json`의
training-normal mean/std를 읽어 공통 spectrogram 정규화를 먼저 적용한다.

```text
log-mel window
-> training-normal mean/std 공통 정규화
-> 3채널 복사
-> 224x224 resize
-> ImageNet mean/std 정규화
-> ResNet18
```

A/B는 공통 정규화에 반드시 같은 `normalization_stats.json`을 사용한다.

## 현재 데이터 구조 확인

실제 feature 파일 예시는 다음 구조다.

```text
project/features/fan_+6dB_16000.npy
shape = (clip, 8, 128, 64)
```

즉 clip 하나당 8개의 window가 이미 만들어져 있다.
현재 저장소의 기존 파일별 loader는 이 묶음 구조와 맞지 않으므로,
다음 단계에서 `manifest.csv`와 feature 배열 index를 연결하는 loader로 수정해야 한다.

## Colab 실행

```python
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/MIMII_project_preprocessed.zip" /content/
!unzip -q /content/MIMII_project_preprocessed.zip -d /content/

!git clone https://github.com/BangHaeun/DeepLearing_ModelB_ResNet-kNN.git
%cd DeepLearing_ModelB_ResNet-kNN
```

결과는 `out_root`에 저장한다.

## 파일

| 파일 | 역할 |
|---|---|
| `config.yaml` | 경로, normalization stats 경로, 실험 조건 |
| `dataset.py` | 데이터 로드, 공통 정규화, clip 단위 split |
| `model_dn2.py` | ResNet18 특징 추출, ImageNet 정규화, kNN 점수 |
| `evaluate.py` | clip 점수, AUROC, 검증 기반 임계치 |
| `run_dn2.py` | 전체 실행 |
| `inspect_data.py` | 데이터 구조 확인 |
