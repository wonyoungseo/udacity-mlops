import pytest
import wandb
import pandas as pd

run = wandb.init('exercise')


@pytest.fixture(scope='session')
def data():
    local_path = run.use_artifact("exercise_5/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path, low_memory=False)

    return df


def test_data_length(data):
    '''
    test on the amount of data received. must be larger than 1000.
    '''
    assert len(data) > 1000


def test_number_of_columns(data):
    '''
    test on the number of columns. must have 19 columns in total.
    '''
    assert data.shape[1] == 19
