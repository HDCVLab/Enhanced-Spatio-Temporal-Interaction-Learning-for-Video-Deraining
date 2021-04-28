## Enhanced Spatio-Temporal Interaction Learning for Video Deraining: A Faster and Better Framework
Kaihao Zhang, Dongxu Li, Wenhan Luo, Wen-Yan Lin, Fang Zhao, Wenqi Ren, Wei Liu, Hongdong Li

### Installation
To replicate the environment:

```bash
cd proj/code
conda install --file requirements.txt
```

### Training
**Please first modify bash files accordingly with your data folder path.**

```bash
cd proj/code/run_scripts
```
(1) Train on NTU dataset:
Put data under /$YOUR_ROOTPATH/derain/NTU-derain
```bash
cd proj/code/run_scripts/
bash train_resnet18_5pic.sh
```

(2) Train on RainSys25 light dataset:
Put data under /$YOUR_ROOTPATH/derain/RainSyn25
```bash
cd proj/code/run_scripts/
bash train_resnet18_rainsys_light_5pic.sh
```

(3) Train on RainSys25 heavy dataset:
Put data under /$YOUR_ROOTPATH/derain/RainSyn25
```bash
cd proj/code/run_scripts/
bash train_resnet18_rainsys_heavy_5pic.sh
```

### Testing with pre-trained weights
**Please first modify bash files accordingly with your data folder path.**

Download checkpoints and put in ```code/best_checkpoints```

(1) Test on NTU dataset:
```bash
cd proj/code/run_scripts/
bash test_ntu_npic.sh
```

(2) Test on RainSys25 light dataset:
```bash
cd proj/code/run_scripts/
bash test_light_npic.sh
```

(3) Test on RainSys25 heavy dataset:
```bash
cd proj/code/run_scripts/
bash test_heavy_npic.sh
```

### Citation
```bibtex
  @article{zhang2021enhanced,
    title={Enhanced Spatio-Temporal Interaction Learning for Video Deraining: A Faster and Better Framework},
    author={Zhang, Kaihao and Li, Dongxu and Luo, Wenhan and Lin, Wen-Yan and Zhao, Fang and Ren, Wenqi and Liu, Wei and Li, Hongdong},
    journal={arXiv preprint arXiv:2103.12318},
    year={2021}
  }
```