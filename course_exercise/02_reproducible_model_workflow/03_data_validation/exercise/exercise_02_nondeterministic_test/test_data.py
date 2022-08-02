import pytest
import wandb
import pandas as pd
import scipy.stats


run = wandb.init(project='exercise_8', job_type='data_tests')

@pytest.fixture(scope='session')
def data():

    local_path = run.use_artifact('exercise_6/data_train.csv:latest').file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact('exercise_6/data_test.csv:latest').file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2



def test_kolmogorov_smirnov(data):

    sample1, sample2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Bonferroni correction for multiple hypothesis testing
    # (see my blog post on this topic to see where this comes from:
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)
    alpha = 0.05
    alpha_prime = 1 - (1 - alpha) ** (1 / len(numerical_columns))

    for col in numerical_columns:

        # 2-sample KS test
        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col], alternative='tow-sided')

        # assertion when p_value > alpha_prime. rejecting null hypothesis(the data is same)
        assert p_value > alpha_prime, f"Columns `{col}` failed test. p_value={p_value}, alpha_prime={alpha_prime}"