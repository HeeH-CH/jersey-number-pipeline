# Conda to UV Migration Summary

## Overview
Migrated the jersey-number-pipeline from Conda environments to UV package manager with separate virtual environments for each component.

## Problems Solved

### 1. CUDA Compatibility with RTX 3090
- **Problem**: PyTorch 1.7.1+cu101 doesn't support RTX 3090 (sm_86 compute capability)
- **Solution**: Upgraded to PyTorch 1.9.0+cu111 which supports sm_86

### 2. PyTorch Lightning hparams Property Conflict
- **Problem**: `AttributeError: can't set attribute 'hparams'` in PyTorch 1.9+
- **Solution**: Changed `self.hparams` to `self._cfg_hparams` in:
  - `reid/centroids-reid/modelling/bases.py`
  - `reid/centroids-reid/train_ctl_model.py`

### 3. NumPy 2.x Incompatibility
- **Problem**: Compiled modules incompatible with NumPy 2.x
- **Solution**: Used NumPy 1.x in UV environments

## UV Environments Created

### 1. Root Environment (`./.venv`)
**Purpose**: Main pipeline execution and shared dependencies
**Location**: `/home/coder/project/jersey-number-pipeline/.venv`
**Python**: 3.9+
**Key Dependencies**:
- torch 1.9.0+cu111
- torchvision
- opencv-python
- pandas
- scipy
- numpy
- tqdm
- pytorch-lightning
- einops

**Configuration**: `pyproject.toml` (root)

### 2. Centroids-ReID Environment
**Purpose**: Person re-identification feature extraction
**Location**: `/home/coder/project/jersey-number-pipeline/reid/centroids-reid/.venv`
**Python**: 3.9+
**Key Dependencies**:
- torch 1.9.0+cu111
- torchvision 0.10.0+cu111
- pytorch-lightning 1.4.0
- einops
- mlflow
- opencv-python
- tqdm
- yacs

**Configuration**: `reid/centroids-reid/pyproject.toml`

### 3. ViTPose Environment
**Purpose**: Human pose estimation
**Location**: `/home/coder/project/jersey-number-pipeline/pose/ViTPose/.venv`
**Python**: 3.8+
**Key Dependencies**:
- torch (via torchvision dependency)
- torchvision
- json-tricks
- matplotlib
- munkres
- numpy
- opencv-python
- pillow
- scipy
- xtcocotools>=1.8
- timm>=0.4
- einops>=0.6

**Removed**: `chumpy` (only used in demo notebooks)
**Configuration**: `pose/ViTPose/pyproject.toml`

### 4. PARSeq Environment
**Purpose**: Scene text recognition for jersey numbers
**Location**: `/home/coder/project/jersey-number-pipeline/str/parseq/.venv`
**Python**: 3.9+
**Key Dependencies**:
- torch 2.8.0 (newer version)
- torchvision
- transformers
- timm
- einops
- tensorboard

**Configuration**: `str/parseq/pyproject.toml` (updated with UV PyTorch sources)

## Files Modified

### Configuration Files
1. **configuration.py**
   - Changed `reid_home = 'reid/'` → `reid_home = 'reid/centroids-reid'`
   - Ensures correct path to ReID venv

2. **main.py**
   - Replaced all conda activation calls with direct UV venv paths
   - ReID: `reid_venv = str(Path(__file__).resolve().parent / config.reid_home / '.venv/bin/python')`
   - Pose: `pose_venv = str(Path(__file__).resolve().parent / config.pose_home / '.venv/bin/python')`
   - STR: `parseq_venv = str(Path(__file__).resolve().parent / config.str_home / '.venv/bin/python')`

### ReID Module
3. **reid/centroids-reid/requirements.txt**
   - Updated torch from 1.7.1+cu101 to 1.9.0+cu111
   - Updated torchvision to 0.10.0+cu111

4. **reid/centroids-reid/modelling/bases.py**
   - Line 64: `self._cfg_hparams = AttributeDict(hparams)` (was `self.hparams`)
   - Lines 71, 74, 77-82, 87, 99-100, etc.: Replaced all `self.hparams` with `self._cfg_hparams`

5. **reid/centroids-reid/train_ctl_model.py**
   - All references to `self.hparams` changed to `self._cfg_hparams`

6. **reid/centroids-reid/pyproject.toml** (newly created)
   - UV configuration with PyTorch index for cu111

### Pose Module
7. **pose/ViTPose/requirements/runtime.txt**
   - Removed `chumpy` dependency

8. **pose/ViTPose/pyproject.toml** (newly created)
   - UV configuration with PyTorch index for cu116

### STR Module
9. **str/parseq/pyproject.toml** (updated)
   - Added UV PyTorch sources configuration

## How to Run

### Main Pipeline
```bash
# Use root venv's python (system python3 doesn't have torch)
/home/coder/project/jersey-number-pipeline/.venv/bin/python main.py SoccerNet test
```

### Individual Components
```bash
# ReID feature extraction
/home/coder/project/jersey-number-pipeline/reid/centroids-reid/.venv/bin/python centroid_reid.py --tracklets_folder <path> --output_folder <path>

# ViTPose
/home/coder/project/jersey-number-pipeline/pose/ViTPose/.venv/bin/python pose.py <config> <checkpoint> --img-root / --json-file <input> --out-json <output>

# PARSeq
cd /home/coder/project/jersey-number-pipeline/str/parseq
.venv/bin/python str.py <model> --data_root=<path> --batch_size=1 --inference --result_file <output>
```

## Environment Management

### Create Environments (if needed)
```bash
# Root environment
uv venv
uv sync

# ReID environment
cd reid/centroids-reid
uv venv
uv sync

# ViTPose environment
cd pose/ViTPose
uv venv
uv sync

# PARSeq environment
cd str/parseq
uv venv
uv sync
```

### Add/Update Dependencies
```bash
cd <environment_directory>
uv add <package>
uv sync
```

## Pipeline Stages

1. **Soccer Ball Filter** - Identifies and filters out soccer ball tracklets
2. **Feature Generation** (~50 min) - ReID feature extraction for 1212 tracklets
3. **Gaussian Filtering** - Removes outliers based on features
4. **Legibility Classification** - Classifies jersey number legibility
5. **Pose Estimation** - ViTPose human pose detection
6. **Crop Generation** - Generates cropped images based on pose
7. **STR Recognition** - PARSeq scene text recognition
8. **Result Combination** - Combines and processes predictions
9. **Evaluation** - Calculates accuracy metrics

## Compatibility Notes

- All environments use CUDA-enabled PyTorch builds
- ReID uses cu111 (CUDA 11.1) for PyTorch 1.9.0 compatibility
- ViTPose and PARSeq use cu116 (CUDA 11.6) for newer PyTorch versions
- Root environment uses cu124 (CUDA 12.4) for PyTorch 2.x

## Migration Date
February 1, 2026
