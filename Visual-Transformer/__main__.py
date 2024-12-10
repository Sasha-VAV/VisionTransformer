import hydra
from pathlib import Path
from omegaconf import DictConfig
from . import Program
from .utils import convert_to_abs_path


# TODO
#  - dvc


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    abspath = str(Path(__file__).parent.parent.resolve()) + "/"
    program = Program(cfg, abspath)
    program.run()  # Run program, options are located in Visual-Transformer/config/config.yaml
    path_to_your_image = (
        "data/img/corgi.jpg"  # Change to the image that you want to classify
    )
    print(f"\n\nLet's classify {path_to_your_image}")
    program(convert_to_abs_path(abspath, path_to_your_image))
    pass


if __name__ == main():
    main()
