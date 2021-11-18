
# End-to-end ML pipeline

## 1. Chain each module

## 2. Train the model through pipeline

```shell
mlflow run --no-conda .
```

## 3. Production Run

- Do a production run by changing the project_name to genre_classification_prod.
- Override parameter to only execute one or more steps.
- Tag the model as `prod`