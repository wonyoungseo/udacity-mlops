import pytest
import wandb
import pandas as pd

run = wandb.init(project="exercise_7", job_type='data_tests')


@pytest.fixture(scope='session')
def data():

    local_path = run.use_artifact("exercise_5/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path, low_memory=False)

    return df


def test_column_presence_and_type(data):
    '''
    test function to verify whether the dataset contains valid columns and type
    '''
    required_columns = {
        "time_signature": pd.api.types.is_integer_dtype,
        "key": pd.api.types.is_integer_dtype,
        "danceability": pd.api.types.is_float_dtype,
        "energy": pd.api.types.is_float_dtype,
        "loudness": pd.api.types.is_float_dtype,
        "speechiness": pd.api.types.is_float_dtype,
        "acousticness": pd.api.types.is_float_dtype,
        "instrumentalness": pd.api.types.is_float_dtype,
        "liveness": pd.api.types.is_float_dtype,
        "valence": pd.api.types.is_float_dtype,
        "tempo": pd.api.types.is_float_dtype,
        "duration_ms": pd.api.types.is_integer_dtype,  # This is integer, not float as one might expect
        "text_feature": pd.api.types.is_string_dtype,
        "genre": pd.api.types.is_string_dtype
    }

    # check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_func in required_columns.items():
        assert format_verification_func(data[col_name]), f"Column {col_name} failed test {format_verification_func}"


def test_class_names_genre(data):
    '''
    test function to verify whether the column contains valid categories
    '''

    known_classes = [
        "Dark Trap",
        "Underground Rap",
        "Trap Metal",
        "Emo",
        "Rap",
        "RnB",
        "Pop",
        "Hiphop",
        "techhouse",
        "techno",
        "trance",
        "psytrance",
        "trap",
        "dnb",
        "hardstyle",
    ]

    assert data['genre'].isin(known_classes).all(), f"Failed test. Following classes are part of known classes : {list(set(data['genre'].unique()) - set(known_classes))}"

def test_column_ranges(data):
    '''
    test function to verify range
    '''

    ranges = {
        "time_signature": (1, 5),
        "key": (0, 11),
        "danceability": (0, 1),
        "energy": (0, 1),
        "loudness": (-35, 5),
        "speechiness": (0, 1),
        "acousticness": (0, 1),
        "instrumentalness": (0, 1),
        "liveness": (0, 1),
        "valence": (0, 1),
        "tempo": (50, 250),
        "duration_ms": (20000, 1000000),
    }

    for col_name, (min_val, max_val) in ranges.items():

        assert data[col_name].dropna().between(min_val, max_val).all(), (
            f"Column {col_name} failed range test. Valid range {min_val} ~ {max_val}, "
            f"instead range={data[col_name].min()}~{data[col_name].max()}"
        )

