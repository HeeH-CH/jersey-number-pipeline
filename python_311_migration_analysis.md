# Python 3.11+ 마이그레이션 심층 분석 보고서

**생성일:** 2026-02-01
**대상 프로젝트:** jersey-number-pipeline
**분석 목적:** Python 3.11 이상으로의 마이그레이션 전략 수립

---

## 1. 개요

본 프로젝트는 스포츠 영상에서 선수 등 번호 인식을 위한 파이프라인으로, 여러 외부 서브모듈과 다양한 딥러닝 프레임워크를 통합하여 사용합니다. 현재 Python 3.9 기반으로 구성되어 있으며, Python 3.11 이상으로 마이그레이션하기 위한 심층 분석을 수행했습니다.

---

## 2. 현재 환경 분석

### 2.1 메인 프로젝트 설정

**파일:** [pyproject.toml](pyproject.toml)

```toml
requires-python = ">=3.9,<3.10"
```

**메인 의존성:**
| 라이브러리 | 현재 버전 | Python 3.11 호환성 |
|-----------|----------|-------------------|
| torch | 1.9.0+cu111 | ⚠️ 문제 있음 |
| torchvision | 0.10.0+cu111 | ⚠️ 문제 있음 |
| pytorch-lightning | 1.6.5 | ❌ 호환되지 않음 |
| torchmetrics | 0.9.3 | ❌ 호환되지 않음 |
| numpy | 1.26.4 | ✅ 호환 |
| opencv-python | 4.10.0.84 | ✅ 호환 |
| pandas | >=2.3.3 | ✅ 호환 |
| scipy | >=1.13.1 | ✅ 호환 |

### 2.2 서브모듈별 환경 설정

#### 2.2.1 PARSeq (Scene Text Recognition)

**위치:** `str/parseq/`
**Conda 환경:** `parseq2` (Python 3.9)
**관련 파일:** [setup.py:110](setup.py#L110)

**주요 의존성:**
| 라이브러리 | 현재 버전 | Python 3.11 호환성 |
|-----------|----------|-------------------|
| torch | 1.13.1+cu117 | ⚠️ 문제 있음 |
| torchvision | 0.14.1+cu117 | ⚠️ 문제 있음 |
| pytorch-lightning | 1.9.5 | ❌ 호환되지 않음 |
| torchmetrics | 1.0.3 | ❌ 호환되지 않음 |
| numpy | 1.25.2 | ✅ 호환 |

#### 2.2.2 Centroids-ReID (Person Re-identification)

**위치:** `reid/centroids-reid/`
**Conda 환경:** `centroids` (Python 3.8)
**관련 파일:** [setup.py:60](setup.py#L60)

**주요 의존성:**
| 라이브러리 | 현재 버전 | Python 3.11 호환성 |
|-----------|----------|-------------------|
| torch | 1.7.1+cu101 | ❌ 매우 구버전 |
| torchvision | 0.8.2+cu101 | ❌ 매우 구버전 |
| pytorch-lightning | 1.1.4 | ❌ 매우 구버전 |

#### 2.2.3 ViTPose (Human Pose Estimation)

**위치:** `pose/ViTPose/`
**Conda 환경:** `vitpose` (Python 3.8)
**관련 파일:** [setup.py:84](setup.py#L84)

**주요 의존성:**
| 라이브러리 | 현재 버전 | Python 3.11 호환성 |
|-----------|----------|-------------------|
| torch | >=1.3 | ✅ 업그레이드 가능 |
| torchvision | (버전 없음) | ✅ 업그레이드 가능 |
| xtcocotools | >=1.8 | ⚠️ 확인 필요 |
| mmcv-full | 1.4.8 | ⚠️ 확인 필요 |

#### 2.2.4 SAM2 (Sharpness-Aware Minimization)

**위치:** `sam2/`
**관련 파일:** [setup.py:141](setup.py#L141)

SAM 옵티마이저 라이브러리로, 별도의 conda 환경 없이 메인 환경에서 사용됩니다.

---

## 3. 주요 호환성 문제 분석

### 3.1 PyTorch 버전 충돌

**문제:** 각 컴포넌트가 서로 다른 PyTorch 버전 사용

| 컴포넌트 | PyTorch 버전 | CUDA 버전 | Python 3.11 지원 |
|---------|-------------|-----------|-----------------|
| 메인 프로젝트 | 1.9.0 | cu111 | ❌ |
| PARSeq | 1.13.1 | cu117 | ⚠️ 제한적 |
| Centroids-ReID | 1.7.1 | cu101 | ❌ |
| ViTPose | >=1.3 | - | ✅ |

**Python 3.11에서 PyTorch 지원:**
- PyTorch 2.0+: Python 3.11 완전 지원
- PyTorch 1.13.x: Python 3.11 제한적 지원
- PyTorch 1.12 이하: Python 3.11 미지원

### 3.2 PyTorch Lightning 호환성

| 현재 버전 | Python 3.11 호환 | 권장 버전 |
|----------|-----------------|-----------|
| 1.1.4 (ReID) | ❌ | 2.x |
| 1.6.5 (메인) | ❌ | 2.x |
| 1.9.5 (PARSeq) | ❌ | 2.x |

PyTorch Lightning 2.x부터 Python 3.11을 공식 지원합니다.

### 3.3 코드 수준 호환성 문제

#### 3.3.1 Deprecated API 사용

**파일:** [centroid_reid.py:5](centroid_reid.py#L5)
```python
import distutils.version  # Python 3.12에서 제거 예정
```

#### 3.3.2 Type Annotation 관련

Python 3.11에서는 `typing` 모듈의 새로운 기능을 활용할 수 있지만, 현재 코드에서는 구버전 스타일의 타입 힌트를 사용합니다.

#### 3.3.3 Conda 환경 분리

**파일:** [setup.py](setup.py)

각 서브모듈이 별도의 conda 환경을 사용합니다:
- `parseq2` (Python 3.9)
- `centroids` (Python 3.8)
- `vitpose` (Python 3.8)

이러한 분리된 환경 구조는 통합된 Python 3.11 환경으로의 마이그레이션을 복잡하게 만듭니다.

---

## 4. Python 3.11 신규 기능 활용 가능성

### 4.1 성능 향상 기능

1. **더 빠른 실행 속도:** Python 3.11은 3.9 대비 약 10-60% 더 빠릅니다
2. **향상된 에러 메시지:** 디버깅 효율성 향상
3. **예외 그룹 (Exception Groups):** 복잡한 에러 처리 개선

### 4.2 타입 시스템 개선

- `Self` 타입
- `TypeAlias` 사용 가능
- 개선된 제너릭 타입 지원

### 4.3 TOML 파싱 내장

`tomllib` 모듈이 표준 라이브러리에 추가되어 별도 의존성 없이 TOML 파일 파싱 가능

---

## 5. 마이그레이션 전략

### 5.1 단계별 접근법

#### 단계 1: 사전 준비 (1-2일)

1. **종속성 매핑**
   - 모든 서브모듈의 의존성 문서화
   - Python 3.11 호환 패키지 버전 확인

2. **테스트 환경 구축**
   - 현재 Python 3.9 환경에서 전체 테스트 수행
   - 베이스라인 성능 지표 기록

#### 단계 2: 의존성 업그레이드 (3-5일)

**권장 버전:**

| 라이브러리 | 현재 | 권장 (Python 3.11) |
|-----------|------|-------------------|
| torch | 1.9.0/1.13.1/1.7.1 | 2.1.0+cu118 |
| torchvision | 0.10.0/0.14.1/0.8.2 | 0.16.0+cu118 |
| pytorch-lightning | 1.6.5/1.9.5/1.1.4 | 2.1.0 |
| torchmetrics | 0.9.3/1.0.3 | 1.2.0 |
| numpy | 1.26.4/1.25.2 | 1.26.0+ |
| opencv-python | 4.10.0.84 | 4.8.0+ |

**업그레이드 우선순위:**
1. PyTorch → 2.1.0 (모든 컴포넌트 통합)
2. PyTorch Lightning → 2.1.0
3. TorchMetrics → 1.2.0

#### 단계 3: 코드 수정 (2-3일)

**필수 수정 사항:**

1. **`distutils.version` 제거**
   ```python
   # 기존
   import distutils.version

   # 수정
   from packaging import version
   ```

2. **Typing 스타일 업데이트** (선택사항)

3. **Conda 환경 구조 재설계**
   - 단일 Python 3.11 환경으로 통합 고려
   - 또는 환경별 분리 유지하되 모두 3.11로 업그레이드

#### 단계 4: 모델 호환성 검증 (2-3일)

**중요:** 사전 훈련된 모델 호환성 확인

1. **모델 weight 로드 테스트**
   - Legibility classifier (ResNet34)
   - PARSeq STR 모델
   - Centroids-ReID 모델
   - ViTPose 모델

2. **PyTorch 버전 간 weight 호환성**
   - PyTorch는 일반적으로 backward compatible
   - 새로운 버전에서 구버전 weight 로드 가능

#### 단계 5: 테스트 및 검증 (3-4일)

1. **단위 테스트**
2. **통합 테스트**
3. **성능 벤치마크**
4. **추론 정확도 검증**

### 5.2 환경 통합 옵션

#### 옵션 A: 단일 환경 통합 (권장)

**장점:**
- 관리 용이성
- 디스크 공간 절약
- 환경 간 전환 불필요

**단점:**
- 의존성 충돌 가능성
- 디버깅 복잡성

**구성:**
```toml
# pyproject.toml
requires-python = ">=3.11"

dependencies = [
    "torch==2.1.0+cu118",
    "torchvision==0.16.0+cu118",
    "pytorch-lightning==2.1.0",
    "torchmetrics==1.2.0",
    # ... 기타 의존성
]
```

#### 옵션 B: 다중 환경 유지 (현재 방식 유지)

**장점:**
- 의존성 격리
- 독립적 컴포넌트 개발 용이

**단점:**
- 관리 복잡성
- 더 많은 디스크 공간

**구성:**
- `jersey-main`: Python 3.11 (메인 프로젝트)
- `parseq2`: Python 3.11 (PARSeq)
- `centroids`: Python 3.11 (ReID)
- `vitpose`: Python 3.11 (ViTPose)

### 5.3 롤백 계획

마이그레이션 실패 시를 대비한 롤백 전략:

1. **Git을 이용한 버전 관리**
   - 마이그레이션 전 브랜치 생성
   - 각 단계별 커밋

2. **Docker 이미지 활용**
   - 현재 Python 3.9 환경 Docker 이미지 저장

3. **Conda 환경 백업**
   ```bash
   conda create --name jersey-pipeline-backup --clone jersey-pipeline
   ```

---

## 6. 잠재적 리스크 및 완화 방안

### 6.1 리스크 매트릭스

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|----------|
| 사전 훈련 모델 호환성 문제 | 중 | 높 | 사전 테스트, weight 변환 |
| CUDA 버전 호환성 | 중 | 높 | Docker 사용, CUDA 확인 |
| 외부 라이브러리 미지원 | 낮 | 중 | 대안 라이브러리 찾기 |
| 성능 저하 | 낮 | 중 | 벤치마킹, 프로파일링 |
| 디버깅 어려움 | 중 | 낮 | 철저한 테스트 |

### 6.2 특정 라이브러리 이슈

#### 6.2.1 xtcocotools

**위치:** ViTPose 의존성
**이슈:** Python 3.11 호환성 확인 필요
**완화:** 최신 버전 사용 또는 포크 버전 확인

#### 6.2.2 mmcv-full

**위치:** ViTPose 의존성
**이슈:** 버전 1.4.8이 구버전
**완화:** mmcv (MMSerialization) 최신 버전으로 업그레이드 고려

#### 6.2.3 gdown

**현재 버전:** >=5.2.1
**호환성:** Python 3.11 지원

---

## 7. 예상 소요 시간

| 단계 | 예상 시간 | 비고 |
|------|----------|------|
| 사전 준비 | 1-2일 | 의존성 조사, 테스트 환경 구축 |
| 의존성 업그레이드 | 3-5일 | PyTorch, 라이브러리 업그레이드 |
| 코드 수정 | 2-3일 | 호환성 문제 수정 |
| 모델 호환성 검증 | 2-3일 | 사전 훈련 모델 테스트 |
| 테스트 및 검증 | 3-4일 | 전체 파이프라인 테스트 |
| **총계** | **11-17일** | 약 2-3주 |

---

## 8. 권장 사항

### 8.1 우선 순위

1. **높음:** PyTorch 2.x로 업그레이드
2. **높음:** PyTorch Lightning 2.x로 업그레이드
3. **중간:** 단일 Python 3.11 환경으로 통합
4. **중간:** `distutils` 제거
5. **낮음:** Python 3.11 신규 기능 활용

### 8.2 Docker 활용

Python 3.11 환경의 Docker 이미지를 구축하여 재현 가능성을 확보하는 것을 권장합니다.

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 3.11 설치
RUN apt-get update && apt-get install -y python3.11 python3.11-venv

# PyTorch 2.1 설치
RUN pip3.11 install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 기타 의존성 설치
...
```

### 8.3 점진적 마이그레이션

전체 시스템을 한 번에 마이그레이션하기보다 컴포넌트별로 순차적으로 마이그레이션하는 것을 권장합니다.

**권장 순서:**
1. SAM2 (가장 간단)
2. PARSeq (독립적)
3. Centroids-ReID
4. ViTPose (가장 복잡)
5. 메인 파이프라인

---

## 9. 결론

본 프로젝트를 Python 3.11 이상으로 마이그레이션하는 것은 기술적으로 가능하지만, 다음과 같은 주요 과제가 있습니다:

1. **PyTorch 버전 통합 필요:** 각 컴포넌트가 서로 다른 PyTorch 버전을 사용
2. **환경 구조 재설계:** 현재 다중 conda 환경 구조 재검토 필요
3. **사전 훈련 모델 호환성:** 모델 weight 호환성 검증 필수
4. **충분한 테스트:** 전체 파이프라인의 정확도와 성능 검증 필요

**권장 접근법:** 점진적 마이그레이션과 철저한 테스트를 통해 안정성을 확보하면서 진행하는 것을 권장합니다. PyTorch 2.x로 업그레이드하면 Python 3.11의 성능 향상 혜택을 최대로 활용할 수 있습니다.

---

## 10. 참고 자료

- [Python 3.11 Release Notes](https://docs.python.org/3.11/whatsnew/3.11.html)
- [PyTorch 2.0 Release Notes](https://pytorch.org/blog/PyTorch-2.0-release/)
- [PyTorch Lightning 2.0 Migration Guide](https://lightning.ai/docs/pytorch/stable/version/2.0.0/upgrade/v2.0_changes.html)
- [What's New in Python 3.11](https://docs.python.org/3.11/whatsnew/3.11.html)

---

**문서 버전:** 1.0
**작성자:** Claude (AI Assistant)
**마지막 업데이트:** 2026-02-01
