import os
import json

import hydra
from omegaconf import DictConfig
import mlflow



@hydra.main(config_name='config')
def go(config: DictConfig):

    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    # get the path at the root of MLflow project
    root_path = hydra.utils.get_original_cwd()

    # Serialize random forest configuration
    model_config = os.path.abspath("random_forest_config.json")
    with open(model_config, "w+") as fp:
        json.dump(dict(config['random_forest']), fp)

    _ = mlflow.run(
        os.path.join(root_path, "model_train_random_forest"),
        "main",
        parameters={
            "train_data": config['data']['train_data'],
            "model_config": model_config
        },
        use_conda=False
    )


if __name__ == "__main__":
    go()