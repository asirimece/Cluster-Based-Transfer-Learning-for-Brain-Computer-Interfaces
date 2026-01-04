import hydra
from omegaconf import DictConfig
from src import base
from src import transfer

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig):
    experiment_type = config.experiment.type
    
    if experiment_type == "base":
        base.run(config)
    elif experiment_type == "transfer":
        transfer.run(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

if __name__ == "__main__":
    main()
