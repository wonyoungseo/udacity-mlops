import tempfile
import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def go(args):

    run = wandb.init(project="exercise_6", job_type="split_data")

    # get artifact from wandb
    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)

    df = pd.read_csv(artifact.file(), low_memory=False)


    # split train/test
    logger.info("Splitting data into train and test")
    splits = {}

    splits['train'], splits['test'] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify != 'null' else None
    )

    # save artifacts.
    # we use a temporary directory so we do not leave any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():

            artifact_name = f"{args.artifact_name_root}_{split}.csv"
            temp_path = os.path.join(tmp_dir, artifact_name)
            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # save then upload to W&B
            df.to_csv(temp_path)
            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}"
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            # wait for the artifact to be uploaded to W&B
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@"
    )

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--artifact_name_root", type=str, required=True)
    parser.add_argument("--artifact_type", type=str, default="raw_data", required=False)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--random_state", type=int, default=42, required=False)
    parser.add_argument("--stratify", type=str, default='null', required=False)

    args = parser.parse_args()
    go(args)