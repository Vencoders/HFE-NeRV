# HFE-NeRV

This repository provides the implementation for the paper: **"High-Frequency Enhanced Hybrid Neural Representation for Video Compression"**.

<!-- Optional: We will release codes coming soon. (如果代码确实还未完全准备好，可以保留此句，否则建议删除) -->

## Quick Start

### Requirements

Install the required dependencies:
```bash
pip install -r requirements.txt
Use code with caution.
Markdown
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
Use code with caution.
Reproducing Experiments
Training
Example command to train a model (e.g., ~1.5M parameters on bunny):
python train_nerv_all.py \
   --outf training_output/bunny_1.5M \ # Specify output folder for checkpoints & logs
   --data_path data/bunny \
   --vid bunny \                     # Video identifier
   --modelsize 1.5 \                 # Target model size (Millions of parameters)
   --conv_type convnext pshuffel \
   --act gelu \
   --norm none \
   --crop_list 640_1280 \
   --resize_list -1 \
   --loss L2 \
   --enc_strds 5 4 4 2 2 \
   --enc_dim 64_16 \
   --dec_strds 5 4 4 2 2 \
   --ks 0_1_5 \
   --reduce 1.2 \
   -e 300 \                          # Number of training epochs
   --eval_freq 30 \                  # Evaluation frequency (epochs)
   --lower_width 12 \
   -b 2 \                            # Batch size
   --lr 0.001                        # Learning rate
Use code with caution.
Bash
Evaluation & Output Generation
Evaluate a trained model and generate reconstructed images/videos:
python train_nerv_all.py \
   --outf evaluation_output/bunny_1.5M \ # Specify output folder for results
   --data_path data/bunny \
   --vid bunny \
   --modelsize 1.5 \                 # Should match the trained model's size
   --conv_type convnext pshuffel \
   --act gelu \
   --norm none \
   --crop_list 640_1280 \
   --resize_list -1 \
   --loss L2 \
   --enc_strds 5 4 4 2 2 \
   --enc_dim 64_16 \
   --dec_strds 5 4 4 2 2 \
   --ks 0_1_5 \
   --reduce 1.2 \
   --lower_width 12 \
   \
   --eval_only \                     # Enable evaluation mode
   --weight checkpoints/hnerv-1.5m-e300.pth \ # IMPORTANT: Path to your trained model checkpoint
   --quant_model_bit 8 \             # Optional: Quantize model weights (bits)
   --quant_embed_bit 6 \             # Optional: Quantize embeddings (bits)
   --dump_images \                   # Optional: Save reconstructed frames
   --dump_videos                     # Optional: Save reconstructed video
