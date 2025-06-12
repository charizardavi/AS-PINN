import hydra
from omegaconf import DictConfig
from src.engine.trainer import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
