'''
mlflow run --no-conda . \
-P input_artifact="exercise_4/genres_mode.parquet:latest" \
-P artifact_name="preprocessed_data.csv" \
-P artifact_type="cleaned_data" \
-P artifact_description="Data after preprocessing"
'''


import os
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)
logger = logging.getLogger()

def perform_drop_duplicates(df):
    logging.info("Exectute Drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def perform_process_missing_val(df):
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']
    return df

def go(args):
    run = wandb.init(project='exercise_5')
    artifact = run.use_artifact(args.input_artifact)
    df = pd.read_parquet(artifact.file())

    df = perform_drop_duplicates(df)
    df = perform_process_missing_val(df)
    filename = "processed_data.csv"
    df.to_csv(filename)

    # add file as wandb Artifact
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(filename)

    # log artifact into wandb
    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # remove the preprocessed file
    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@"
    )

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--artifact_name", type=str, required=True)
    parser.add_argument("--artifact_type", type=str, required=True)
    parser.add_argument("--artifact_description", type=str, required=True)
    args = parser.parse_args()

    go(args)