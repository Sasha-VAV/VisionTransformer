# Visual transformer, based 
***
## You can import this module via
```shell
pip install VisualTransformer
```
## How to use it?
1. Configure config yaml file to work with model
2. Module has 2 methods:
   1. run - use for training and testing model
   2. \_\_call__ - use to classify images and utilize model
Example of the code:
```python
import VisualTransformer as ViT
import hydra
from pathlib import Path


@hydra.main(config_path="path_to_your_config", config_name="name_of_your_config")
def main(cfg):
    abspath = str(Path(__file__).parent.resolve()) + "/"  # Get absolute path, if your config works with relative paths
    model = ViT.Program(cfg, abspath)  # Create object of class Program to work with transformer

    model.run()  # Run config, that you wrote to train and test model

    arr = ["path1", "path2", "path3"]
    print(model(arr, False))  # Utilize model


if __name__ == "__main__":
    main()
```
***
## Credits
Thanks for the idea and pretrained models to:
- A huge credit to my teacher, [Alexandr Korchemnyj](https://github.com/Yessense)
- Idea is based on science article
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- I have used pretrained model from SWAG, [check the license](https://github.com/facebookresearch/SWAG/blob/main/LICENSE)
