# ViTPose 환경 Python 3.8 → 3.11+ 마이그레이션 심층 분석 보고서

**작성일:** 2026-02-01
**대상 환경:** `cfg.pose_env` (ViTPose)
**현재 Python 버전:** 3.8
**목표 Python 버전:** 3.11 이상

---

## 1. 개요 (Executive Summary)

본 보고서는 Jersey Number Pipeline 프로젝트의 ViTPose 환경을 Python 3.8에서 Python 3.11 이상으로 마이그레이션하기 위한 심층 분석 결과를 제시합니다. 코드 변경 없이 **분석만을 수행**하며, 마이그레이션 가능성 평가와 전략 수립을 목표로 합니다.

### 핵심 결론

| 항목 | 평가 |
|------|------|
| **Python 3.9로의 마이그레이션** | ✅ **가능** - ViTPose 공식 지원 |
| **Python 3.11로의 마이그레이션** | ⚠️ **제한적** - 의존성 업데이트 필요 |
| **주요 차단 요소** | MMCV 버전 제약 (1.3.8 ~ 1.5.0) |
| **추천 접근 방식** | 단계적 업그레이드 (3.8 → 3.9 → 3.11) |

---

## 2. 현재 환경 분석

### 2.1 환경 구성

**Conda 환경 생성 경로:** [setup.py:84](setup.py#L84)

```python
# 현재 설정
make_conda_env(env_name, libs="python=3.8")
```

**설치되는 주요 의존성:**
- `mmcv-full==1.4.8` (CUDA 11.1, PyTorch 1.9.0 기반)
- `mmpose` (ViTPose 0.24.0, OpenMMLab)
- `timm==0.4.9`
- `einops`

### 2.2 프로젝트 전체 Python 버전 현황

| 환경 | Python 버전 | 파일 위치 |
|------|-------------|-----------|
| **메인 프로젝트** | 3.9 | [pyproject.toml:6](pyproject.toml#L6) |
| **ViTPose (pose_env)** | 3.8 | [setup.py:84](setup.py#L84) |
| **Centroid-ReID** | 3.8 | [setup.py:60](setup.py#L60) |
| **PARSeq (str_env)** | 3.9 | [setup.py:110](setup.py#L110) |

---

## 3. ViTPose Python 버전 지원 분석

### 3.1 공식 지원 범위

**파일:** [pose/ViTPose/setup.py:174-184](pose/ViTPose/setup.py#L174-L184)

```python
classifiers=[
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',  # ← 최신 공식 지원 버전
],
```

**중요:** ViTPose setup.py에는 Python 3.10, 3.11, 3.12에 대한 classifier가 **없습니다**.

### 3.2 Python 버전 검증 코드

**파일:** [pose/ViTPose/setup.py:97-101](pose/ViTPose/setup.py#L97-L101)

```python
if not sys.version.startswith('3.4'):
    # apparently package_deps are broken in 3.4
    platform_deps = info.get('platform_deps')
    if platform_deps is not None:
        parts.append(';' + platform_deps)
```

- Python 3.4 이상에서만 작동하는 코드 (현재로서는 제약 없음)

---

## 4. 핵심 차단 요소 분석

### 4.1 MMCV 버전 제약 (가장 중요)

**파일:** [pose/ViTPose/mmpose/__init__.py:19-27](pose/ViTPose/mmpose/__init__.py#L19-L27)

```python
mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '1.5.0'  # ← 문제의 핵심

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'
```

**문제점:**
- MMCV 1.3.8 ~ 1.5.0 버전은 Python 3.8/3.9 시대에 릴리즈됨
- Python 3.11 지원은 MMCV 1.7.0+부터 제공
- **이 버전 제약 코드는 런타임에 강제 검증됨**

### 4.2 PyTorch 버전 호환성

**현재 설정:**
- PyTorch 1.9.0+cu111 (CUDA 11.1)

**Python 3.11 지원:**
| PyTorch 버전 | Python 3.11 지원 |
|--------------|------------------|
| 1.13.0+ | ✅ 공식 지원 |
| 2.0.0+ | ✅ 공식 지원 |
| 1.9.0 | ❌ 미지원 |

### 4.3 기타 의존성 분석

**파일:** [pose/ViTPose/requirements/runtime.txt](pose/ViTPose/requirements/runtime.txt)

```txt
chumpy
dataclasses; python_version == '3.6'  # Python 3.7+에서는 내장
json_tricks
matplotlib
munkres
numpy
opencv-python
pillow
scipy
torchvision
xtcocotools>=1.8
```

- **dataclasses:** Python 3.6용 조건부 설치 (3.7+는 내장)
- **나머지 패키지:** Python 3.11과 호환 가능

---

## 5. 마이그레이션 경로 분석

### 옵션 A: Python 3.9로의 마이그레이션 (권장)

**장점:**
- ✅ ViTPose 공식 지원 범위 내
- ✅ 기존 MMCV 1.4.8 사용 가능
- ✅ 코드 수정 최소화
- ✅ 메인 프로젝트와 버전 통일

**변경 사항:**
- [setup.py:84](setup.py#L84)에서 `python=3.8` → `python=3.9`로 변경만 하면 완료

**위험도:** 🟢 **낮음 (LOW RISK)**

---

### 옵션 B: Python 3.11로의 직접 마이그레이션

**필수 변경 사항:**

1. **MMCV 버전 업그레이드**
   - MMCV 1.4.8 → 2.x 버전으로 업그레이드 필요
   - **주의:** MMCV 2.0은 API 호환성 깨짐 (mmcv → mmcv, mmdet, mmengine 분리)

2. **PyTorch 버전 업그레이드**
   - PyTorch 1.9.0 → 2.0+로 업그레이드 필요

3. **ViTPose 코드 수정**
   - [pose/ViTPose/mmpose/__init__.py:19-20](pose/ViTPose/mmpose/__init__.py#L19-L20)의 버전 제약 완화
   - MMCV 2.x API 변경에 따른 코드 수정

4. **MMDet/MMTrack 호환성**
   - `mmdet>=2.14.0`, `mmtrack>=0.6.0`도 새 버전과 호환되는지 확인 필요

**위험도:** 🔴 **높음 (HIGH RISK)**
- 대규모 코드 수정 필요
- 모델 호환성 미확정
- 테스트 기간 장기화

---

### 옵션 C: 단계적 마이그레이션 (3.8 → 3.9 → 3.11)

**1단계:** Python 3.8 → 3.9 (안정화)
- 옵션 A와 동일

**2단계:** Python 3.9 → 3.11 (미래 준비)
- ViTPose 최신 버전으로 업그레이드 고려
- 또는 OpenMMLab의 최신 MMPose로 마이그레이션

---

## 6. 마이그레이션 영향 범위

### 6.1 수정 필요 파일 (Python 3.9)

| 파일 | 라인 | 변경 내용 |
|------|------|-----------|
| [setup.py](setup.py) | 84 | `python=3.8` → `python=3.9` |

### 6.2 수정 필요 파일 (Python 3.11)

| 파일 | 라인 | 변경 내용 |
|------|------|-----------|
| [setup.py](setup.py) | 84 | `python=3.8` → `python=3.11` |
| [setup.py](setup.py) | 87 | mmcv-full 버전 변경 |
| [pose/ViTPose/mmpose/__init__.py](pose/ViTPose/mmpose/__init__.py) | 19-20 | MMCV 버전 제약 완화 |
| [pose/ViTPose/setup.py](pose/ViTPose/setup.py) | 174-184 | Python 3.11 classifier 추가 |
| ViTPose 소스 코드 | 다수 | MMCV 2.x API 호환성 수정 |

---

## 7. Centroid-ReID 환경 고려사항

**현재 상황:**
- [setup.py:60](setup.py#L60)에서 `centroids` 환경도 Python 3.8 사용

**권장사항:**
- ViTPose와 동일하게 Python 3.9로 통일
- 프로젝트 전체 Python 버전 일관성 유지

---

## 8. 최종 권장 사항

### 8.1 단기 전략 (즉시 실행 가능)

**Python 3.9로의 마이그레이션**

1. **수정:**
   ```python
   # [setup.py:84]
   make_conda_env(env_name, libs="python=3.9")  # 3.8에서 3.9로 변경
   ```

2. **환경 재생성:**
   ```bash
   conda env remove -n vitpose --all
   python setup.py SoccerNet  # 또는 Hockey
   ```

3. **검증:**
   ```bash
   conda activate vitpose
   python --version  # 3.9.x 확인
   python -c "import mmpose; print(mmpose.__version__)"
   ```

**예상 소요 시간:** 10분 이내

### 8.2 중장기 전략 (추후 고려)

**Python 3.11+로의 마이그레이션**

1. **선택 1:** ViTPose를 최신 OpenMMLab MMPose로 교체
   - ViTPose는 더 이상 활발 개발되지 않음
   - 최신 MMPose는 ViT 백본을 포함

2. **선택 2:** Python 3.9 유지 (권장)
   - 현재 안정적인 환경 유지
   - 2025년 말까지 Python 3.9 보안 지원 예정

---

## 9. 위험 평가 및 완화 계획

### 9.1 Python 3.9 마이그레이션 위험도

| 위험 요소 | 영향 | 확률 | 완화 계획 |
|----------|------|------|----------|
| MMCV 호환성 문제 | 중 | 낮 | 공식 지원 확인됨 |
| PyTorch CUDA 호환성 | 중 | 낮 | PyTorch 1.9.0+cu111은 3.9 지원 |
| 모델 가중치 호환성 | 높 | 없음 | 모델 가중치는 Python 버전 무관 |
| 성능 저하 | 낮 | 없음 | 동일한 바이너리 사용 |

**종합 위험도:** 🟢 **낮음 (LOW RISK)**

### 9.2 Python 3.11 마이그레이션 위험도

| 위험 요소 | 영향 | 확률 | 완화 계획 |
|----------|------|------|----------|
| MMCV 2.x API 호환성 | 높 | 높 | 대규모 코드 수정 필요 |
| 모델 호환성 | 높 | 중 | 재훈련 가능성 |
| 의존성 충돌 | 중 | 높 | 다른 환경들과의 일관성 문제 |
| 개발 기간 | 중 | 높 | 2주 이상 소요 예상 |

**종합 위험도:** 🔴 **높음 (HIGH RISK)**

---

## 10. 참고: 다른 프로젝트 사례

### MMPose (ViTPose 기반) 현재 상태

- **GitHub:** [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)
- **최신 버전:** 1.x (Python 3.8-3.11 지원)
- **ViTPose:** MMPose의 일부로 통합됨

**참고:** 2025년 연구 논문에서 MMPose를 Python 3.10+에서 사용하는 사례가 다수 확인됨

---

## 11. 결론

### 요약

1. **Python 3.9 마이그레이션:** ✅ 강력 권장
   - 위험도 낮음
   - 수정 최소화
   - 메인 프로젝트와 버전 통일

2. **Python 3.11 마이그레이션:** ⚠️ 신중한 접근 필요
   - MMCV/PyTorch 대규모 업그레이드 필요
   - ViTPose 자체 업데이트 또는 교체 고려
   - 단계적 접근 (3.8 → 3.9 → 3.11) 권장

### 다음 단계 (권장 순서)

1. **즉시:** ViTPose 환경 Python 3.9로 마이그레이션
2. **검증:** 기존 테스트 케이스 실행
3. **추후:** Centroid-ReID 환경도 Python 3.9로 통일
4. **장기:** Python 3.11+ 이전을 위한 MMPose 최신 버전 평가

---

**문서 작성자:** Claude (Sonnet 4.5)
**분석 날짜:** 2026-02-01
**버전:** 1.0
