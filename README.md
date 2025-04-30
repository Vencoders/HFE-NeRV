# HFE-NeRV

This repository provides the implementation for the paper: **"High-Frequency Enhanced Hybrid Neural Representation for Video Compression"**.

<!-- Optional: We will release codes coming soon. -->

## Quick Start

### Requirements

Install the required dependencies:
```bash
pip install -r requirements.txt

Dataset Preparation
Before training, download the bunny, UVG, and DAVIS validation datasets. Organize your dataset folder as follows (the default path expected by scripts is ./data/, or you can specify --data_path):
./
├── data/
│   ├── bunny/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── UVG_Full/
│   │   ├── Beauty_1920x1080_120/
│   │   │   ├── 001.png
│   │   │   ├── 002.png
│   │   │   └── ...
│   │   ├── Bosphorus_1920x1080_120/
│   │   └── ...
│   └── DAVIS/
│       └── JPEGImages/
│           └── 1080p/
│               ├── blackswan/
│               │   ├── 00000.jpg
│               │   ├── 00001.jpg
│               │   └── ...
│               ├── bmx-trees/
│               └── ...
├── train_nerv_all.py
└── requirements.txt

