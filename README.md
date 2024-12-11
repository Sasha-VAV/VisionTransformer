# Transformer neural network, that choose between dog and cat
***
## How to launch it?
- I highly recommend to use conda env with python 310, you can get conda [here](https://www.anaconda.com/download)
```shell
conda create -n conda310 python=3.10
```
- Then you need to activate it
```shell
conda activate conda310
```
- Configure python interpreter, I highly recommend to use python 3.10, 
but you can edit Visual-Transformer/\_\_main__.py and remove raising version compatibility error
```shell
pip install -r requirements.txt
```
- You need to install poetry
```shell
conda install poetry
```
- Load all the libs
```shell
poetry install
```
- Download huggingface_hub library
```shell
pip install huggingface_hub
```
- Download weights and default images from huggingface
```shell
python setup.py
```
- Download dvc
```shell
pip install dvc
```
- Install weights and default images
```shell
dvc pull
```
- Run module
```shell
python -m Visual-Transformer
```

## How to configure it?
- Move to Visual-Transformer/config/config.yaml, read inline documentation and configure it

## Current stats:
- Architecture: Visual Transformer
- Train set size: 24000 images
- Test set size: 1000 images
- Max achieved **accuracy**: 99.7

## Max accuracy research
- default LeNet: 74.1
- default AlexNet: 93.9
- AlexNet + data augmentation: 95.6
- Visual Transformer + data augmentation: 99.7

***
## Credits
Thanks for the idea and pretrained models to:
- A huge credit to my teacher, [Alexandr Korchemnyj](https://github.com/Yessense)
- Idea is based on science article
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- I have used pretrained model from SWAG, [check the license](https://github.com/facebookresearch/SWAG/blob/main/LICENSE)
