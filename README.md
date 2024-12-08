# Transformer neural network, that choose between dog and cat
***
## How to launch it?
- Run command
```shell
poetry install
```
- Install wandb
```shell
pip install wandb
```
- You need to train model,
or [download weights](https://huggingface.co/Sashavav/dogs_vs_cats_vit/tree/main/dogs_vs_cats4.0/pretrained_configs)
- Extract downloaded folder in the root folder
- To replace default photos and set yours,
you should move to dogs_vs_cats4/dvc4_config/config.py and change
list_of_images_paths parameter
- Configure conda 310 env
- Launch dogs_vs_cats4/\_\_main\_\_.py

## How to configure it?
- Move to dogs_vs_cats4/dvc4_config_config.py and
read all inline comments, then change all the data

## Current stats:
- Architecture: Visual Transformer
- Train set size: 24000 images
- Test set size: 1000 images
- Max achieved **accuracy**: 99.6

## Max accuracy research
- default LeNet: 74.1
- default AlexNet: 93.9
- AlexNet + data augmentation: 95.6
- Visual Transformer + data augmentation: 99.6

***
## Credits
Thanks for the idea and pretrained models to:
- A huge credit to my teacher, [Alexandr Korchemnyj](https://github.com/Yessense)
- Idea is based on science article
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- I have used pretrained model from SWAG, [check the license](https://github.com/facebookresearch/SWAG/blob/main/LICENSE)
