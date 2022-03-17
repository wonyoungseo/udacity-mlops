# DVC pipeline


## Direction

- The `prepare.py` preprocesses the fake data and the second trains a logistic regression. Take these scripts and create two DVC stages, one for each script. 
- Specify the dependencies and outputs for each.
- Additionally, the hyperparameter for logistic regression is hard coded. Change it to read from a param.yaml and include the paramter in the stage.


## Create `params.yaml`

```yaml
# params.yaml

train:
  test_size: 0.25
  random_state: 24
  C: 1.0
```

## Command

### Prepare stage

```bash
dvc run -n prepare \
        -d prepare.py \
        -d fake_data.csv \
        -o X.csv \
        -o y.csv \
        python ./prepare.py
```

### Train stage

```bash
dvc run -n train \
        -d train.py \
        -d X.csv \
        -d y.csv \
        -p train \
        python ./train.py
```


## Output after creating pipeline version

After running the command, `dvc.yaml` file is created.

```yaml
# dvc.yaml

stages:
  prepare:
    cmd: python ./prepare.py
    deps:
    - fake_data.csv
    - prepare.py
    outs:
    - X.csv
    - y.csv
  train:
    cmd: python ./train.py
    deps:
    - X.csv
    - train.py
    - y.csv
    params:
    - train
```