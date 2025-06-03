# HNeRV: High-Frequency Enhanced Hybrid Neural Representation for video compression  
### [Paper](https://arxiv.org/pdf/2411.06685) | [Project Page](https://vencoders.github.io/) | [UVG Data](http://ultravideo.fi/#testsequences) 


[//]: # ([Hao Chen]&#40;https://haochen-rye.github.io&#41;,)

[//]: # (Matthew Gwilliam,)

[//]: # (Ser-Nam Lim,)

[//]: # ([Abhinav Shrivastava]&#40;https://www.cs.umd.edu/~abhinav/&#41;<br>)
This is the official implementation of the paper "High-Frequency Enhanced Hybrid Neural Representation for video compression".


## Get started
We run with Python 3.8, you can set up a conda environment with all dependencies like so:
```
pip install -r requirements.txt 
```

## Reproducing experiments

### Training FEHNeRV
HNeRV of 1.5M is specified with ```'--modelsize 1.5'```, and we balance parameters with ```'-ks 0_1_5 --reduce 1.2' ```
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```


### Evaluation & dump images and videos
To evaluate pre-trained model, use ```'--eval_only --weight [CKT_PATH]'``` to evaluate and specify model path. \
For model and embedding quantization, use ```'--quant_model_bit 8 --quant_embed_bit 6'```.\
To dump images or videos, use  ```'--dump_images --dump_videos'```.
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2  \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001 \
   --eval_only --weight checkpoints/hnerv-1.5m-e300.pth \
   --quant_model_bit 8 --quant_embed_bit 6 \
    --dump_images --dump_videos
```

### Video inpainting
We can specified inpainting task with ```'--vid bunny_inpaint_50'``` where '50' is the mask size.
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny_inpaint_50   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```

## Citation
If you find our work useful in your research, please cite:
```
@article{YU2025127552,
title = {High-Frequency Enhanced Hybrid Neural Representation for video compression},
journal = {Expert Systems with Applications},
volume = {281},
pages = {127552},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.127552},
author = {Li Yu and Zhihui Li and Jimin Xiao and Moncef Gabbouj},
}
```

[//]: # (## Contact)

[//]: # (If you have any questions, please feel free to email the authors: chenh@umd.edu)
