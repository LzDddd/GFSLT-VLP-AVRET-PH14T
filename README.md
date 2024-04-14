This repo is the implementation of [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP) with AVRET on RWTH-PHOENIX-Weather 2014T dataset. Thanks to their great work. 
It currently includes code and pretrained features for the gloss-free SLT task.

## Installation

```bash
conda create -n gfslt python==3.8
conda activate gfslt

# Please install PyTorch according to your CUDA version.
pip install -r requirements.txt
```

## Getting Started

### Preparation
* The pretrain_models of MBart can download from [here](https://pan.baidu.com/s/1x6Dl_uuEp_Y8dFoGS-QhNg). The extract code is `4h2w`. It includes MBart_trimmed and mytran.

* The pretrained features of RWTH-PHOENIX-Weather 2014T can download from [here](https://pan.baidu.com/s/19wXiNXtFpC2RGPxNBzIkkA). The extract code is `ekjj`. And then, put it in the `data/features/` folder.

### Train
```bash
python train_slt.py 
```

## Note
Since GFSLT-VLP is based on the MBart model, we did not apply the local clip self-attention (LCSA) module to it.

# LICENSE
The code is released under the MIT license.