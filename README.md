# MIMII DN2 (ResNet18 + kNN) 이상탐지

## Colab 실행
```python
from google.colab import drive
drive.mount('/content/drive')

# 데이터 (Drive -> 로컬 디스크로 복사 후 압축 해제; Drive에서 바로 읽으면 느림)
!cp "/content/drive/MyDrive/MIMII_project_preprocessed.zip" /content/
!unzip -q /content/MIMII_project_preprocessed.zip -d /content/

# 코드
!git clone https://github.com/<계정>/<저장소>.git
%cd <저장소>

!python inspect_data.py /content/MIMII_project_preprocessed   # 구조 확인 -> config.yaml의 layout 수정
!python run_dn2.py --config config.yaml
```
결과는 `out_root`(Drive)에 저장되며, 끊겨도 다시 실행하면 완료된 조건은 건너뜁니다.

## 파일
| 파일 | 역할 |
|---|---|
| `config.yaml` | 경로와 실험 조건 |
| `dataset.py` | .npy 로드, 클립 단위 학습/검증/테스트 분할 |
| `model_dn2.py` | ResNet18 특징 추출, kNN 점수 |
| `evaluate.py` | 클립 점수, AUROC, 검증 기반 임계치 |
| `run_dn2.py` | 전체 실행 (특징 캐시 -> 점수 csv -> 결과 csv) |
| `inspect_data.py` | 데이터 폴더 구조 확인 |

## 실험 바꾸는 법 (`config.yaml`)
Exp2 `snrs`, Exp3 `srs`, Exp4 `layers`, 부록 `k`만 바꿔 다시 실행.
