# Jersey Number Pipeline – Full Guide

End-to-end pipeline for jersey number recognition in sports videos and images.  
This repository integrates **pose estimation**, **re-identification**, **legibility classification**, and **scene text recognition (STR)** into a single research pipeline.

Supported datasets:
- **SoccerNet** (tracklet-level inference)
- **Hockey** (image-level inference)

---

## 1. Repository Overview

This repository is **not a standalone model**.  
It is an **orchestrated research pipeline** that combines multiple external repositories and pretrained models.

Core characteristics:
- Step-based pipeline (selective execution)
- External repos mounted inside fixed subdirectories
- Supports both **training** and **inference**
- Designed for reproducible research, not lightweight deployment

---

## 2. Directory Structure

jersey-number-pipeline/
├── data/ # Datasets (SoccerNet, Hockey)
├── models/ # Pretrained model weights
├── experiments/ # Training outputs / checkpoints
├── pose/ # Pose estimation (ViTPose)
│ └── ViTPose/ # External repo (must be cloned)
├── reid/ # Re-identification
│ └── centroids-reid/ # External repo (must be cloned)
├── sam/ # Segment Anything (optional but supported)
├── str/
│ └── parseq/ # PARSeq STR (included version)
│
├── main.py # Pipeline entry point (orchestrator)
├── configuration.py # Paths and configuration
├── setup.py # Auto setup script
│
├── legibility_classifier.py # Jersey readability classifier
├── number_classifier.py # Jersey number classification logic
├── centroid_reid.py # ReID-based filtering
├── gaussian_outliers.py # Outlier removal (Gaussian fitting)
├── pose.py # Pose-guided ROI cropping
├── str.py # Scene text recognition wrapper
├── jersey_number_dataset.py # Dataset loader
├── networks.py # Model definitions
├── helpers.py # Shared utilities
└── README.md


---

## 3. Environment Setup

### 3.1 Python Environment

Recommended:
```bash
python >= 3.9
Create virtual environment:

python -m venv .venv
source .venv/bin/activate
Install dependencies:

pip install -r requirements.txt
If requirements.txt is missing, dependencies are installed via setup.py.

4. Clone External Dependencies
This repository expects other repositories to exist at fixed paths.

4.1 Automatic Setup (Recommended)
python setup.py
This script:

Clones required repositories

Places them in correct directories

Downloads pretrained checkpoints (if available)

4.2 Manual Setup (If Needed)
Pose Estimation – ViTPose
cd pose
git clone https://github.com/ViTAE-Transformer/ViTPose.git
Place checkpoint inside:

pose/ViTPose/checkpoints/
Re-Identification – centroids-reid
cd reid
git clone https://github.com/mkoshkina/centroids-reid.git
Download pretrained weights into:

reid/centroids-reid/models/
Scene Text Recognition – PARSeq
Already included at:

str/parseq/
(Optional) Segment Anything
git clone https://github.com/facebookresearch/segment-anything.git sam
5. Dataset Setup
5.1 SoccerNet
Download SoccerNet jersey dataset and place it under:

data/SoccerNet/
Expected structure (simplified):

SoccerNet/
├── images/
├── tracklets/
├── annotations/
5.2 Hockey Dataset
Place hockey images under:

data/Hockey/
6. Configuration
Edit configuration.py to set:

Dataset paths

Model checkpoints

Output directories

Example:

DATA_ROOT = "./data"
MODEL_ROOT = "./models"
EXPERIMENT_ROOT = "./experiments"
7. Pipeline Architecture
7.1 Hockey (Image-Level Pipeline)
Legibility Classification

Determines if jersey number is readable

Scene Text Recognition (STR)

Recognizes number string using PARSeq

Flow:

Image → Legibility Classifier → STR → Jersey Number
7.2 SoccerNet (Tracklet-Level Pipeline)
ReID feature extraction

Outlier removal (Gaussian fitting)

Legibility classification

Pose-guided ROI cropping

Scene Text Recognition (STR)

Tracklet-level consolidation

Flow:

Tracklet Frames
  → ReID filtering
  → Legibility filtering
  → Pose-guided crop
  → STR per frame
  → Tracklet aggregation
8. Running Inference
8.1 SoccerNet (Tracklet Level)
python main.py SoccerNet test
8.2 Hockey (Image Level)
python main.py Hockey test
9. Training
9.1 Train Legibility Classifier (Hockey)
python legibility_classifier.py \
  --train \
  --data data/Hockey \
  --output experiments/legibility
9.2 Train STR (PARSeq Fine-tuning)
Hockey:

python main.py Hockey train --train_str
SoccerNet:

python main.py SoccerNet train --train_str
10. main.py – Action System
main.py contains an action list, allowing selective execution of steps.

Example actions:

reid

pose

legibility

str

consolidate

This allows:

Debugging individual stages

Ablation experiments

Partial pipeline execution

11. Key Files Explained
File	Purpose
main.py	Pipeline orchestrator
legibility_classifier.py	Readability classification
pose.py	Pose-guided ROI cropping
str.py	Scene text recognition
centroid_reid.py	ReID-based filtering
gaussian_outliers.py	Noise removal
configuration.py	Global config
setup.py	Dependency setup
12. Research-Oriented Design Notes
Not optimized for real-time inference

Heavy reliance on pretrained components

Best suited for:

Academic experiments

Benchmark reproduction

Pipeline ablation studies

13. Common Issues
Missing checkpoints
Ensure all pretrained weights are placed in:

models/
pose/ViTPose/checkpoints/
reid/centroids-reid/models/
CUDA issues
Ensure PyTorch CUDA version matches your GPU driver.

14. Citation
If you use this repository in research, cite the original paper associated with this work.

15. License
Refer to individual sub-repositories for license terms.


---

### 다음 단계 제안
원하면 다음도 가능하다:

1. **`main.py` action 흐름 다이어그램 (Mermaid)**
2. **실제 SoccerNet 트랙릿 통합 로직 코드 해설**
3. **`-1 label` 문제 포함, 현재 구조의 한계 분석**
4. **프로덕션용으로 경량화한 리팩토링 구조 제안**

원하는 방향을 말해주면 그에 맞게 이어서 정리하겠다.