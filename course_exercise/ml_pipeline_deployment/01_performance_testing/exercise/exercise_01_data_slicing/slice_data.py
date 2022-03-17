import pandas as pd
from sklearn import datasets


def load_iris_dataframe():
    """
    function to load iris dataset from scikit learn library and convert into pandas dataframe format
    """
    iris = datasets.load_iris()
    feature_names = [f.replace('(cm)', '').strip() for f in iris['feature_names']]
    df = pd.DataFrame(iris['data'], columns=feature_names)
    df['class'] = iris['target']

    return df


def slice_dataset(df: pd.DataFrame, class_col: str, target_feature: str):
    """
    function for calculating descriptive stats on slices of the iris dataset
    """

    for cls in df[class_col].unique():
        df_tmp = df[df[class_col]==cls].copy()

        mean = df_tmp[target_feature].mean()
        std = df_tmp[target_feature].std()

        print(f"Class: {cls}")
        print(f"{target_feature} mean: {mean:.4f}")
        print(f"{target_feature} std: {std:.4f}")
    print("\n")


if __name__ == "__main__":

    df = load_iris_dataframe()

    slice_dataset(df, "class", "sepal length")
    slice_dataset(df, "class", "sepal width")
    slice_dataset(df, "class", "petal length")
    slice_dataset(df, "class", "petal width")

