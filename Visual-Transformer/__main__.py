import hydra
from pathlib import Path
from omegaconf import DictConfig
from . import Program


# TODO
#  - dvc
#  - requirements


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    abspath = str(Path(__file__).parent.parent.resolve()) + "\\"
    program = Program(cfg, abspath)
    program.run()
    pass


if __name__ == main():
    main()
