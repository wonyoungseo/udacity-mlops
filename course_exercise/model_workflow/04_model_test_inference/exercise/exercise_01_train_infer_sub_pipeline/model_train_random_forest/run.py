import argparse
import logging
import json

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(project="exercise_10", job_type="train")

    logger.info("Donwloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    logger.info("Setting up pipeline")
    pipe = get_inference_pipeline(args)

    logger.info("Fitting")
    pipe.fit(X_train, y_train)

    logger.info("Scoring")
    score = roc_auc_score(
        y_val, pipe.predict_proba(X_val), average="macro", multi_class="ovo"
    )

    run.summary['AUC'] = score


    # We collect the feature importance for all non-nlp features first
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]

    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) :])

    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")

    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)

    fig_feat_imp.tight_layout()

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_val,
        y_val,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )



def get_inference_pipeline(args):
    '''
    fucntion to build pipeline
    - catgorical features
    - numerical features
    - one for textual ("nlp") features
    '''


    # 1. categorical processing
    categorical_features = sorted(['time_signature', 'key'])
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0), OrdinalEncoder()
    )

    # 2. numerical preprocessing pipeline
    numeric_features = sorted([
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ])

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler()
    )


    # 3. textual ("nlp") preprocessing pipeline
    # This trick is needed because SimpleImputer wants a 2d input, but
    # TfidfVectorizer wants a 1d input. So we reshape in between the two steps
    nlp_features = ["text_feature"]
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(binary=True)
    )


    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("nlp1", nlp_transformer, nlp_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform (i.e., we don't use)
    )

    # Get the configuration for the model
    with open(args.model_config) as fp:
        model_config = json.load(fp)
    # Add it to the W&B configuration so the values for the hyperparams are tracked
    wandb.config.update(model_config)


    # CREATE a Pipeline instances with 2 steps: one step called "preprocessor" using the preprocessor instance, and another one called "classifier" using RandomForestClassifier(**model_config)
    # Append classifier to preprocessing pipeline. Now we have a full prediction pipeline.
    # NOTE: here you should create the Pipeline object directly, and not make_pipeline
    # HINT: Pipeline(steps=[("preprocessor", instance1), ("classifier", LogisticRegression)]) creates a
    #       Pipeline with two steps called "preprocessor" and "classifier" using the sklearn instances instance1
    pipe = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config))
        ]
    )
    return pipe




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a JSON file containing the configuration for the random forest",
        required=True,
    )

    args = parser.parse_args()

    go(args)
