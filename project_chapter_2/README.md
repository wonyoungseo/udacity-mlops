# Chp 2.

## Install requirements

```bash
pip install -r requirements.txt
```

### Test W&B

```bash
source activate udacity
echo "wandb test" > wandb_test
wandb artifact put -n testing/artifact_test wandb_test
```

### Test MLflow

```
mlflow --help
```