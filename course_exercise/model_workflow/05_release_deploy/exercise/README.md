
# End-to-end ML pipeline

## 1. Chain each module

## 2. Train the model through pipeline

```shell
mlflow run --no-conda .
```

## 3. Production Run

- Do a production run by changing the project_name to genre_classification_prod.
    ```shell
    mlflow run --no-conda . -P hydra_options="main.project_name='genre_classification_prod'"
    ```
- Override parameter to only execute one or more steps.
- Tag the model as `prod`

## 4. Release pipeline for production

- fix the config for `prod` 
- commit push to repository
- release the repository

## 5. Run pipeline in MLflow env

- `mlflow run -v 1.0.0 [URL of your Github repo]`
- Q. Is this work in private repository? Need to try it out.


## 6. Deploy MLflow pipeline
