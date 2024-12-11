import hydra
from pathlib import Path
from omegaconf import DictConfig
from . import Program
from .utils import convert_to_abs_path


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Remove next 3 lines if you are sure, that this program is going to work on your interpreter
    import sys
    if sys.version_info[0:2] != (3, 10):
        raise Exception('Requires python 3.10')
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
