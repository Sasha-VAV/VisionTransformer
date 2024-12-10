# Transformer neural network, that choose between dog and cat
***
## How to launch it?
- Configure python interpreter
```shell
pip install -r requirements.txt
```
- Load all the libs
```shell
poetry install
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
- Move to dogs_vs_cats4/dvc4_config_config.py and
read all inline comments, then change all the data

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
