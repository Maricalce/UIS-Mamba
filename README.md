# [ACM MM 2025] UIS-Mamba: Exploring Mamba for Underwater Instance Segmentation
[Runmin Cong<sup><span>1,</span></sup>](), [Zongji Yu<sup><span>1,</span></sup>](), [Hao Fang<sup><span>1,†</span></sup>](), [Haoyan Sun<sup><span>1,</span></sup>](), [Sam Kwong<sup><span>2</span></sup>]()  
<sup><span>†</span></sup> Corresponding author  
<sup>1</sup> School of Control Science and Engineering, Shandong University, Jinan, Shandong, China  
<sup>2</sup> School of Data Science, Lingnan University, Hong Kong, China  
<a href='https://arxiv.org/pdf/2508.00421v1'><img src='https://img.shields.io/badge/ArXiv-2508.00421v1-red'></a> 
<a href='https://github.com/Maricalce/UIS-Mamba'><img src='https://img.shields.io/badge/GitHub-UIS--Mamba-green'></a>


## 📖 Abstract
Underwater Instance Segmentation (UIS) is critical for underwater complex scene detection, but faces challenges like color distortion, blurred boundaries, and complex backgrounds. Mamba, with linear complexity and global receptive fields, is suitable for long-sequence feature tasks, yet its fixed-patch scanning and unfiltered hidden state limit performance in underwater scenes.  

We propose **UIS-Mamba**—the first Mamba-based underwater instance segmentation model—equipped with two core modules:  
1. **Dynamic Tree Scan (DTS)**: Maintains instance internal feature continuity via dynamic patch offset and scaling, guiding minimum spanning tree construction.  
2. **Hidden State Weaken (HSW)**: Suppresses complex background interference using Ncut-based hidden state weakening, focusing information flow on instances.  

UIS-Mamba achieves state-of-the-art (SOTA) performance on UIIS and USIS10K datasets while keeping parameters and computational complexity low.


## 🔍 Core Innovations
### 1. Dynamic Tree Scan (DTS) Module
Addresses fixed-patch scanning limitations in underwater scenes through two steps:  
- **Adaptive Graph Deformation**: Predicts offset (Δx, Δy) and scale (Δw, Δh) parameters to adjust patches, preserving same-instance pixel regions even under color distortion.  
- **Dynamic Graph Pruning**: Fuses spatial distance and semantic similarity to calculate edge weights, generating a Minimum Spanning Tree (MST) to maintain instance semantic integrity.  

### 2. Hidden State Weaken (HSW) Module
Mitigates background interference in Mamba’s hidden state updates:  
- **Ncut-Based Patch Categorization**: Uses MST from DTS to partition patches into foreground (instance) and background, reducing computational complexity.  
- **Hidden State Weaken**: Applies a suppression weight (φ=0.7, optimized via experiments) to background patches during state updates, enhancing instance feature focus.  


## 📊 Experimental Results
### Underwater Instance Segmentation (UIIS Dataset)
| Method          | Backbone       | Params | mAP  | AP₅₀ | AP₇₅ |
|-----------------|----------------|--------|------|------|------|
| WaterMask R-CNN | ResNet-50      | 54M    | 26.4 | 43.6 | 28.8 |
| **UIS-Mamba-T** | UIS-Mamba-T    | 56M    | 29.4 | 46.7 | 31.3 |
| WaterMask R-CNN | ResNet-101     | 67M    | 27.2 | 43.7 | 29.3 |
| **UIS-Mamba-S** | UIS-Mamba-S    | 76M    | 30.4 | 48.6 | 33.2 |
| USIS-SAM        | ViT-H          | 700M   | 29.4 | 45.0 | 32.3 |
| **UIS-Mamba-B** | UIS-Mamba-B    | 115M   | 31.2 | 49.1 | 34.5 |

### Underwater Salient Instance Segmentation (USIS10K Dataset)
| Method          | Backbone       | Params | Class-Agnostic mAP | Multi-Class mAP |
|-----------------|----------------|--------|--------------------|-----------------|
| WaterMask R-CNN | ResNet-50      | 67M    | 58.3               | 37.7            |
| **UIS-Mamba-T** | UIS-Mamba-T    | 56M    | 62.2               | 42.1            |
| WaterMask R-CNN | ResNet-101     | 67M    | 59.0               | 38.7            |
| **UIS-Mamba-S** | UIS-Mamba-S    | 76M    | 63.1               | 44.5            |
| USIS-SAM        | ViT-H          | 701M   | 59.7               | 43.1            |
| **UIS-Mamba-B** | UIS-Mamba-B    | 115M   | 63.8               | 46.2            |


## 🛠️ Environment Setup
### Prerequisites
- Python 3.9+  
- PyTorch 1.13.1+cu117 or higher  
- MMDetection (for detection/segmentation heads)  

### Installation Steps
```bash
# Create conda environment
conda create -n uis-mamba python=3.9
conda activate uis-mamba

# Install PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install -r requirements.txt

```


## 🚀 Train & Evaluate
### 1. Dataset Preparation
Download the two benchmark datasets and organize them as follows:  
```
data/
├── UIIS/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   └── val/
│       ├── images/
│       └── annotations/
└── USIS10K/
    ├── train/
    ├── val/
    └── test/
```
- **UIIS Dataset**: [Official Link](https://github.com/LiamLian0727/WaterMask)  
- **USIS10K Dataset**: [Official Link](https://github.com/LiamLian0727/USIS10K)  

### 2. Training
Run training scripts for UIIS (instance segmentation) or USIS10K (salient instance segmentation):  
```bash
# Train UIS-Mamba on UIIS/USIS10K (1 GPU)
cd tools && python train.py

```

### 3. Evaluation
Evaluate pre-trained models on validation/test sets:  
```bash
# Evaluate on UIIS/USIS10K val set
cd tools && python test.py
```


## 📦 Model Zoo
Pre-trained weights for UIS-Mamba variants are available for download:  

| Model           | Backbone       | Dataset   | mAP  | Params | Download Link                                                                 |
|-----------------|----------------|-----------|------|--------|-------------------------------------------------------------------------------|
| UIS-Mamba-T     | UIS-Mamba-T    | UIIS      | 29.4 | 56M    | [ckpt](https://github.com/Maricalce/UIS-Mamba/releases/tag/v1.0/uis_mamba_t_uiis.pth) |
| UIS-Mamba-S     | UIS-Mamba-S    | UIIS      | 30.4 | 76M    | [ckpt](https://github.com/Maricalce/UIS-Mamba/releases/tag/v1.0/uis_mamba_s_uiis.pth) |
| UIS-Mamba-B     | UIS-Mamba-B    | UIIS      | 31.2 | 115M   | [ckpt](https://github.com/Maricalce/UIS-Mamba/releases/tag/v1.0/uis_mamba_b_uiis.pth) |
| UIS-Mamba-T     | UIS-Mamba-T    | USIS10K   | 42.1 | 56M    | [ckpt](https://github.com/Maricalce/UIS-Mamba/releases/tag/v1.0/uis_mamba_t_usis10k.pth) |
| UIS-Mamba-S     | UIS-Mamba-S    | USIS10K   | 44.5 | 76M    | [ckpt](https://github.com/Maricalce/UIS-Mamba/releases/tag/v1.0/uis_mamba_s_usis10k.pth) |
| UIS-Mamba-B     | UIS-Mamba-B    | USIS10K   | 46.2 | 115M   | [ckpt](https://github.com/Maricalce/UIS-Mamba/releases/tag/v1.0/uis_mamba_b_usis10k.pth) |


## ⭐ BibTeX
If you use UIS-Mamba in your research, please cite our paper:  
```bibtex
@article{cong2025uis,
  title={UIS-Mamba: Exploring Mamba for Underwater Instance Segmentation via Dynamic Tree Scan and Hidden State Weaken},
  author={Cong, Runmin and Yu, Zongji and Fang, Hao and Sun, Haoyan and Kwong, Sam},
  journal={arXiv preprint arXiv:2508.00421},
  year={2025}
}
```


## ❤️ Acknowledgement
This work is supported by:  
- Taishan Scholar Project of Shandong Province (tsqn202306079)  
- National Natural Science Foundation of China (62471278)  
- Research Grants Council of Hong Kong (STG5/E-103/24-R)  

Code is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) and [GrootV](https://github.com/EasonXiao-888/MambaTree).


## ☑️ LICENSE
The code is released under the [MIT License](https://opensource.org/license/MIT).


