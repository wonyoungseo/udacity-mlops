import os

import hydra
import mlflow
from omegaconf import DictConfig


@hydra.main(config_name = 'config')
def go(config: DictConfig):

    # Setup the wandb experiment
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']


    # get path at the root of the MLflow project --> ??
    root_path = hydra.utils.get_original_cwd()


    _ = mlflow.run(
        os.path.join(root_path, 'component'),
        "main",
        parameters={
            'a': config['parameters']['a'],
            'b': config['parameters']['b'],
        },
    )


if __name__ == "__main__":
    go()
