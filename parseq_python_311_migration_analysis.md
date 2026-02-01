# PARSeq Python 3.11 마이그레이션 분석 보고서

**대상 컴포넌트:** Scene Text Recognition (PARSeq)
**위치:** `./str/parseq/`
**Conda 환경:** `cfg.str_env` (parseq2)
**생성일:** 2026-02-01

---

## 1. 개요

PARSeq는 이 프로젝트의 Scene Text Recognition (STR) 컴포넌트로, 선수 등 번호 텍스트를 인식하는 데 사용됩니다. 현재 Python 3.9 환경에서 구동되며 PyTorch 1.13.1 기반입니다.

---

## 2. 현재 환경 분석

### 2.1 환경 설정

**설정 파일:** [setup.py:110](setup.py#L110)
```python
make_conda_env(env_name, libs="python=3.9")
```

**Conda 환경명:** `parseq2`

### 2.2 Python 버전 요구사항

**파일:** [str/parseq/pyproject.toml](str/parseq/pyproject.toml)
```toml
requires-python = ">=3.8"
```

### 2.3 핵심 의존성 분석

| 라이브러리 | 현재 버전 | Python 3.11 호환성 | 비고 |
|-----------|----------|-------------------|------|
| torch | 1.13.1+cu117 | ⚠️ 제한적 | 2.0+ 권장 |
| torchvision | 0.14.1+cu117 | ⚠️ 제한적 | 0.15+ 권장 |
| pytorch-lightning | 1.9.5 | ❌ 미지원 | 2.0+ 필요 |
| torchmetrics | 1.0.3 | ❌ 미지원 | 1.2+ 필요 |
| numpy | 1.25.2 | ✅ 호환 | |
| pillow | 10.0.0 | ✅ 호환 | |
| opencv-python | 4.8.0.76 | ✅ 호환 | |
| timm | 0.9.5 | ✅ 호환 | |
| hydra-core | 1.3.2 | ✅ 호환 | |
| omegaconf | 2.3.0 | ✅ 호환 | |
| tensorboardx | 2.6.2.2 | ✅ 호환 | |

---

## 3. 주요 호환성 문제

### 3.1 PyTorch Lightning ⚠️ 핵심 문제

**현재 버전:** 1.9.5
**문제:** Python 3.11을 공식 지원하지 않음

**PyTorch Lightning Python 지원 현황:**
| 버전 | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|------|-----------|-------------|-------------|-------------|
| 1.9.x | ✅ | ✅ | ❌ | ❌ |
| 2.0.x | ✅ | ✅ | ✅ | ❌ |
| 2.1.x+ | ✅ | ✅ | ✅ | ✅ |

**해결 방안:** PyTorch Lightning 2.1+로 업그레이드 필요

### 3.2 PyTorch 버전

**현재:** 1.13.1+cu117
**권장:** 2.1.0+cu118 이상

PyTorch 2.0부터 Python 3.11을 완전히 지원합니다. PyTorch 1.13.x는 제한적으로 지원하지만, 2.0+로 업그레이드하는 것을 권장합니다.

### 3.3 코드 수준 호환성

분석 결과, 현재 PARSeq 코드베이스는 Python 3.11과 호환되는 현대적인 Python 스타일을 사용합니다:

**호환되는 패턴:**
- 최신 타입 힌트 사용: `from typing import Optional, Callable, Sequence, Tuple`
- f-string 사용
- `pathlib.PurePath` 사용

**문제 없는 코드:**
```python
# strhub/models/parseq/system.py
from typing import Sequence, Any, Optional
from torch import Tensor
```

---

## 4. 모델 호환성 분석

### 4.1 사전 훈련된 모델

**파일 경로:** [configuration.py:84-87](configuration.py#L84-L87)

```python
'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',
'str_model_url': "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
```

**중요:** PyTorch는 backward compatible하므로:
- PyTorch 1.13.1에서 저장한 체크포인트를 PyTorch 2.x에서 로드 가능
- 모델 아키텍처는 변경되지 않으므로 weight 호환성 보장

**테스트 필요:**
```python
# str.py:264
model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
```

### 4.2 Fine-tuned 모델

프로젝트에서 fine-tuning한 모델들도 PyTorch 2.x에서 호환됩니다:
- SoccerNet fine-tuned: `parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt`
- Hockey fine-tuned: `parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt`

---

## 5. 마이그레이션 전략

### 5.1 권장 버전

| 라이브러리 | 현재 | 권장 (Python 3.11) | 변경 사항 |
|-----------|------|-------------------|----------|
| torch | 1.13.1+cu117 | 2.1.0+cu118 | 주요 업그레이드 |
| torchvision | 0.14.1+cu117 | 0.16.0+cu118 | 주요 업그레이드 |
| pytorch-lightning | 1.9.5 | 2.1.0+ | 주요 업그레이드 |
| torchmetrics | 1.0.3 | 1.2.0+ | 호환성 업데이트 |

### 5.2 업그레이드 단계

#### 단계 1: PyTorch Lightning 2.x 마이그레이션

PyTorch Lightning 1.x에서 2.x로의 변경 사항:

**주요 API 변경:**
```python
# 1.x (구버전)
from pytorch_lightning.utilities.types import STEP_OUTPUT

# 2.x (신버전) - 호환됨
from pytorch_lightning.utilities.types import STEP_OUTPUT
```

**필수 수정:**
1. `training_step` 반환 타입 확인
2. `log_dict` → `log`로 변경된 부분 확인
3. GPU 설정 방식 변경 확인

#### 단계 2: requirements 파일 업데이트

**파일:** [str/parseq/requirements/core.cu117.txt](str/parseq/requirements/core.cu117.txt)

```txt
# 변경 전
torch==1.13.1+cu117
torchvision==0.14.1+cu117
pytorch-lightning==1.9.5
torchmetrics==1.0.3

# 변경 후 (Python 3.11)
--extra-index-url https://download.pytorch.org/whl/cu118

torch==2.1.0+cu118
torchvision==0.16.0+cu118
pytorch-lightning==2.1.0
torchmetrics==1.2.0
```

#### 단계 3: setup.py 수정

**파일:** [setup.py:110](setup.py#L110)

```python
# 변경 전
make_conda_env(env_name, libs="python=3.9")

# 변경 후
make_conda_env(env_name, libs="python=3.11")
```

#### 단계 4: configuration.py 업데이트 (선택사항)

**파일:** [configuration.py:4](configuration.py#L4)

```python
# Conda 환경명 변경 (선택사항)
str_env = 'parseq2'  # 또는 'parseq311'
```

---

## 6. PyTorch Lightning 2.x 마이그레이션 가이드

### 6.1 Breaking Changes

#### 6.1.1 `training_step` 시그니처

**기존 (1.x):**
```python
def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
    images, labels = batch
    # ...
    return loss
```

**2.x에서도 호환됨:** 위 코드는 2.x에서도 정상 작동합니다.

#### 6.1.2 `log` 메서드

**기존 (1.x):**
```python
self.log('loss', loss)
```

**2.x:** 동일하게 작동합니다.

#### 6.1.3 `Trainer` API

**1.x에서 2.x로의 주요 변경:**
```python
# 1.x
trainer = pl.Trainer(gpus=1)

# 2.x (권장)
trainer = pl.Trainer(accelerator='gpu', devices=1)

# 또는 자동 감지
trainer = pl.Trainer(accelerator='auto')
```

**PARSeq 코드 영향:** 현재 코드에서는 `Trainer`를 직접 사용하지 않으므로 영향 없음.

### 6.2 PARSeq 코드 호환성 검토

**핵심 파일 분석:**

1. **[str/parseq/strhub/models/parseq/system.py](str/parseq/strhub/models/parseq/system.py)**
   - `training_step`: 2.x 호환 ✅
   - `forward`: 2.x 호환 ✅
   - `encode`, `decode`: 2.x 호환 ✅

2. **[str/parseq/strhub/data/module.py](str/parseq/strhub/data/module.py)**
   - `LightningDataModule`: 2.x 호환 ✅

**결론:** 코드 수정 없이 PyTorch Lightning 2.x로 업그레이드 가능합니다.

---

## 7. 테스트 계획

### 7.1 단위 테스트

1. **체크포인트 로드 테스트**
```python
model = load_from_checkpoint('parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt')
```

2. **추론 테스트**
```python
# str.py 테스트
python3 main.py SoccerNet test --pipeline '{"str": true}'
```

3. **Fine-tuning 테스트**
```python
# str/parseq/train.py 테스트
python3 main.py SoccerNet train --train_str
```

### 7.2 통합 테스트

전체 파이프라인에서 PARSeq 테스트:
```bash
# SoccerNet 데이터셋
python3 main.py SoccerNet test

# Hockey 데이터셋
python3 main.py Hockey test
```

### 7.3 성능 비교

| 메트릭 | Python 3.9 | Python 3.11 | 비고 |
|--------|-----------|-------------|------|
| 추론 속도 | baseline | +10-20% 예상 | Python 3.11 성능 향상 |
| 정확도 | baseline | 동일 예상 | 모델은 동일 |

---

## 8. 롤백 계획

### 8.1 환경 백업

```bash
# 현재 환경 백업
conda create --name parseq2-backup --clone parseq2
```

### 8.2 Git 관리

```bash
# 마이그레이드 전 브랜치 생성
git checkout -b backup/parseq-python39

# 마이그레이션 브랜치
git checkout -b feature/parseq-python311
```

---

## 9. 예상 소요 시간

| 작업 | 예상 시간 |
|------|----------|
| 환경 백업 | 0.5시간 |
| requirements 파일 수정 | 0.5시간 |
| setup.py 수정 | 0.5시간 |
| conda 환경 재생성 | 1시간 |
| 체크포인트 로드 테스트 | 0.5시간 |
| 추론 테스트 | 1시간 |
| 통합 테스트 | 1시간 |
| **총계** | **약 5시간** |

---

## 10. 권장 사항

### 10.1 우선 순위

1. **높음:** PyTorch Lightning 2.x로 업그레이드 (Python 3.11 지원 필수)
2. **높음:** PyTorch 2.x로 업그레이드 (성능 향상)
3. **중간:** TorchMetrics 최신 버전으로 업그레이드
4. **낮음:** Python 3.11 신규 기능 활용

### 10.2 주의사항

1. **CUDA 버전 호환성:** cu117 → cu118으로 변경 시 NVIDIA 드라이버 확인 필요
2. **LMDB 호환성:** 데이터셋 포맷(LMDB)은 Python 버전과 무관하므로 문제 없음
3. **체크포인트 호환성:** PyTorch는 backward compatible하므로 안전

### 10.3 추가 이점

Python 3.11로 업그레이드 시:
- **성능 향상:** Python 3.11은 3.9 대비 약 10-60% 더 빠름
- **메모리 효율:** 향상된 메모리 관리
- **디버깅:** 개선된 에러 메시지

---

## 11. 결론

PARSeq 컴포넌트를 Python 3.11로 마이그레이션하는 것은 **기술적으로 가능하며 비교적 간단**합니다.

**주요 변경 사항:**
1. PyTorch 1.13.1 → 2.1.0
2. PyTorch Lightning 1.9.5 → 2.1.0
3. TorchMetrics 1.0.3 → 1.2.0
4. Python 3.9 → 3.11

**코드 수정:** 최소한 (주로 requirements 파일)

**리스크:** 낮음
- 모델 호환성 보장 (PyTorch backward compatible)
- 코드는 이미 최신 Python 스타일 사용

**예상 이점:**
- Python 3.11 성능 향상 (10-20%)
- PyTorch 2.x의 성능 최적화 (compile, sdpa 등)
- 최신 라이브러리 생태계 활용

---

## 12. 실행 계획 (요약)

```bash
# 1. 기존 환경 백업
conda create --name parseq2-backup --clone parseq2

# 2. Python 3.11 환경 생성
conda create -n parseq2 python=3.11 -y
conda activate parseq2

# 3. PyTorch 2.x 설치
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 4. 기타 의존성 설치
pip install pytorch-lightning==2.1.0 torchmetrics==1.2.0
pip install -r str/parseq/requirements/core.cu117.txt  # (수정된 버전)

# 5. 테스트
python3 main.py Hockey test
```

---

**문서 버전:** 1.0
**작성자:** Claude (AI Assistant)
**마지막 업데이트:** 2026-02-01
