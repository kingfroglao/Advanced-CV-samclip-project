# Talking to SAM: Language-Guided Open-Vocabulary Segmentation

This project implements a **language-guided image segmentation system**. By providing a natural language prompt (e.g., “a black dog on the grass”), the system leverages the **Segment Anything Model (SAM)** and **CLIP** to automatically segment the corresponding object from the image.

---

## Project Overview

This project explores two pipelines that combine natural language with SAM’s segmentation capability:

### 1. SAM + CLIP Ranking

- **SAM (Automatic Mask Generator)** generates all possible candidate masks in the image.  
- **CLIP** extracts both the text prompt features and the features of each masked image region.  
- Compute cosine similarity between text and image features to rank all masks.  
- Return the highest-scoring mask as the segmentation result.

### 2. SAM + Grounding DINO Seeding

- **Grounding DINO** first predicts a bounding box in the image based on the text prompt.  
- The predicted box is used as a **spatial prompt** for **SAM (Sam Predictor)**.  
- SAM generates a high-quality segmentation mask guided by the box.

---

## Installation

### 1) Clone the repository

```bash
git clone [your-repo-URL]
cd language-guided-sam
```

### 2) Create and activate a Python virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3) Install dependencies

First, install **PyTorch** according to your CUDA version from the official PyTorch website.

Then, install project dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt (estimated from project_outline.md):**
```text
torch
torchvision
numpy
opencv-python-headless
matplotlib
tqdm
# SAM
segment-anything
# CLIP
git+https://github.com/openai/CLIP.git
# Grounding DINO
git+https://github.com/IDEA-Research/GroundingDINO.git
transformers
# (other dependencies if needed)
```

---

## Dataset

This project uses the following datasets for evaluation:

- **RefCOCO / RefCOCO+ / RefCOCOg** — benchmarks for referring expression segmentation.  
- **PhraseCut** — a large-scale dataset for phrase-based segmentation.

Place downloaded datasets under the `data/` directory, or modify paths in `src/dataloaders.py` accordingly.

---

## How to Use

### 1) Run the demo

Test a single image using the demo script.

**Method 1: CLIP Ranking**
```bash
python scripts/run_demo.py   --image_path "path/to/your/image.jpg"   --text_prompt "a black dog on the grass"   --method "clip_ranking"   --output_path "path/to/output/mask.png"
```

**Method 2: Grounding DINO Seeding**
```bash
python scripts/run_demo.py   --image_path "path/to/your/image.jpg"   --text_prompt "a black dog on the grass"   --method "dino_seeding"   --output_path "path/to/output/mask.png"
```

---

## Evaluation

Compute metrics such as **mIoU** and **Success@IoU** on standard datasets (e.g., RefCOCO validation split).

**Evaluate CLIP Ranking**
```bash
python scripts/evaluate.py   --dataset "refcoco"   --dataset_split "val"   --method "clip_ranking"
```

**Evaluate Grounding DINO Seeding**
```bash
python scripts/evaluate.py   --dataset "refcoco"   --dataset_split "val"   --method "dino_seeding"
```

---

## Project Structure

```
project-root/
│
├── data/                 # Datasets
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── src/                  # Core source code
│   ├── dataloaders.py    # Data loaders
│   ├── models.py         # Wrappers for SAM, CLIP, Grounding DINO
│   ├── pipeline.py       # Implementations of the two segmentation pipelines
│   └── utils.py          # Utilities (IoU, visualization, etc.)
│
├── scripts/              # Executable scripts
│   ├── evaluate.py       # Evaluation script
│   └── run_demo.py       # Demo script
│
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Summary

This project demonstrates how to combine **SAM**, **CLIP**, and **Grounding DINO** to achieve **language-guided open-vocabulary segmentation**. It supports both ranking-based and detection-guided pipelines, enabling flexible and interpretable multimodal segmentation experiments.
