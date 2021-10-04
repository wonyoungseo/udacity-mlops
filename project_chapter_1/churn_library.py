"""
This file is a set of functions for customer churn predction project

Auther: Wonyoung Seo
Date: Sep 27, 2021
"""

import os
import json
from typing import List
import logging

import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sns.set()


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
)


def read_json(file_path: str) -> dict:
    """
    function to read json file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data: dict = json.load(f)
    return data


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv

    output:
            data: pandas dataframe
    """
    data = pd.read_csv(pth)
    return data


class EncoderHelper:
    '''
    helper class for encoding given columns
    '''

    @staticmethod
    def encode_churn_target_var(
        dataframe: pd.DataFrame, target_col: str, target_val: str, response: str
    ) -> pd.DataFrame:
        """
        function to convert churn column into target variable

        inputs:
                dataframe: pandas dataframe.
                target_var_col: string.
                reponse: string.

        output:
                dataframe: pandas dataframe
        """
        assert (
            target_val in dataframe[target_col].unique()
        ), f"Value '{target_val}' does not exist in target column '{target_col}'"
        dataframe[response] = dataframe[target_col].apply(
            lambda val: 0 if val == target_val else 1)
        return dataframe

    @staticmethod
    def encode_cat_col(dataframe, category_lst, response) -> pd.DataFrame:
        """
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                dataframe: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for
                          naming variables or index y column]

        output:
                dataframe: pandas dataframe with new columns for
        """

        for cat in category_lst:

            encoded_lst = []
            group = dataframe.groupby(cat).mean()[response]
            for val in dataframe[cat]:
                encoded_lst.append(group.loc[val])
            dataframe[f"{cat}_{response}"] = encoded_lst

        return dataframe


def perform_feature_engineering(
    dataframe, X_cols: List[str], y_col: str, test_size: float = 0.3
):
    """
    function to prepare data for model training and split them into train, test dataset

    input:
              dataframe: pandas dataframe
              X_cols: list of columns. contains independent variables/features
              y_cols: string. target variable
              test_size: ratio of test set out of total dataset. default 0.3

    output:
              data_train: X training data
              data_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    y = dataframe[y_col]
    data = dataframe[X_cols].copy()

    data_train, data_test, y_train, y_test = train_test_split(
        data, y, test_size=test_size, random_state=42
    )

    return data_train, data_test, y_train, y_test


class PlotGenerator:
    """
    Wrapper class for generating plots
    """

    def __init__(self, plot_file_path: str):
        self.plot_path = plot_file_path

    def _save_plot(self, plt_object, prefix: str, plot_type: str):
        """
        function to save the generated plot

        input:
                plt_object:
                prefix: str
                plot_type: str

        output:
                None
        """
        file_name = f"{prefix}_{plot_type}.png"
        file_path = os.path.join(self.plot_path, file_name)
        plt_object.savefig(file_path, bbox_inches="tight")
        plt_object.close()
        logging.info(f"Plot saved '{file_path}'")

    def plot_hist(self, dataframe, col_name):
        """
        creates and store histogram plot on given column
        """
        plt.figure()
        ax = plt.gcf()
        dataframe[col_name].hist()
        self._save_plot(plt, col_name, plot_type="histogram")

    def plot_bar(self, dataframe, col_name):
        """
        creates and store bar plot on given column
        """
        plt.figure()
        ax = plt.gcf()
        dataframe[col_name].value_counts("normalize").plot(kind="bar")
        self._save_plot(plt, col_name, plot_type="bar")

    def plot_dist(self, dataframe, col_name):
        """
        creates and store distribution plot on given column
        """
        plt.figure()
        ax = plt.gcf()
        sns.distplot(dataframe[col_name])
        self._save_plot(plt, col_name, plot_type="distribution")

    def plot_corr_heatmap(self, dataframe, color_map="Dark2_r"):
        """
        creates and store heatmap on given data
        """
        plt.figure()
        ax = plt.gcf()
        sns.heatmap(dataframe.corr(), annot=False, cmap=color_map, linewidths=2)
        self._save_plot(plt, "corr", plot_type="heatmap")

    def plot_classification_report(
        self, model_name, y_train, y_train_preds, y_test, y_test_preds
    ):
        """
        creates classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds: training predictions
                y_test_preds: test predictions

        output:
                 None
        """

        plt.rc("figure", figsize=(7, 7))

        plt.text(
            0.01,
            1.25,
            str(f"{model_name} Train"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.7,
            str(classification_report(y_train, y_train_preds)),
            {"fontsize": 10},
            fontproperties="monospace",
        )  # approach improved by OP -> monospace!

        plt.text(
            0.01,
            0.6,
            str(f"{model_name} Test"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.05,
            str(classification_report(y_test, y_test_preds)),
            {"fontsize": 10},
            fontproperties="monospace",
        )  # approach improved by OP -> monospace!
        plt.axis("off")

        self._save_plot(plt, model_name, "classification_report")

    def plot_aucroc_curve(self, model_name, model_object, data_test, y_test):
        """
        creates and stores the AUC-ROC curve of the model

        inputs:
            model_name: str
            model_object: trained model object
            data_test: test dataset
            y_test: test target variable

        output:
            None
        """
        plt.figure()
        ax = plt.gcf()
        plot = plot_roc_curve(model_object, data_test, y_test, alpha=0.8)
        self._save_plot(plt, model_name, "roc_curve")

    def plot_shap_explainer(
            self,
            model_name,
            model_object,
            data_test,
            cv_perform):
        """
        creates and stores the shap values per feature

        inputs:
            model_name: str
            model_object: trained model object
            data_test: test dataset
            cv_perform: bool. indicates whether cross validation was perform

        output:
            None
        """
        plt.figure(figsize=(20, 5))
        ax = plt.gcf()
        if cv_perform:
            explainer = shap.TreeExplainer(model_object.best_estimator_)
        else:
            explainer = shap.TreeExplainer(model_object)
        shap_values = explainer.shap_values(data_test)
        shap.summary_plot(
            shap_values,
            data_test,
            plot_type="bar",
            show=False,
            axis_color="#000000")
        self._save_plot(plt, model_name, "shap")

    def plot_feature_importance(
            self,
            model_name,
            model_object,
            data_train,
            cv_perform):
        """
        creates and stores the feature importances

        inputs:
            model_name: str
            model_object: trained model object
            data_train: train dataset
            cv_perform: bool. indicates whether cross validation was perform

        output:
            None
        """

        # Calculate feature importance
        if cv_perform:
            importance = model_object.best_estimator_.feature_importances_
        else:
            importance = model_object.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importance)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [data_train.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel("Importance")

        # Add bars
        plt.bar(range(data_train.shape[1]), importance[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(data_train.shape[1]), names, rotation=90)

        self._save_plot(plt, model_name, "feature_importance")


class EDAHelper(PlotGenerator):
    """
    perform eda on dataframe and save figures to images folder
    """

    def __init__(self, plot_dir):
        super().__init__(plot_dir)

    @staticmethod
    def return_data_summary(dataframe: pd.DataFrame):
        """
        function to return statistics of the data

        input:
                dataframe: pandas DataFrame
        ouput:
                None
        """
        print("Data Shape:")
        dataframe.info()
        print("\n")
        print("Statistics per column:")
        print(dataframe.describe())


class ChurnModel(PlotGenerator):
    """
    Model object for churn model.
    """

    def __init__(self, model_dir: str, plot_dir: str, random_state: int = 42):
        super().__init__(plot_dir)
        model_filename_format = "{}_model.pkl"
        self._model_path_format = os.path.join(
            model_dir, model_filename_format)
        self.random_state = random_state

    def train(
            self,
            model_code: str,
            model_config: dict,
            data):
        """
        train and store model and model performance record

        input:
                model_type: str
                model_config: dic
                data: tuple (data_train, data_test, y_train, y_test)
        output:
                None
        """

        data_train = data[0]
        data_test = data[1]
        y_train = data[2]
        y_test = data[3]

        model = self._init_model(model_code, self.random_state)
        logging.info(
            f"{model_config['model_name']} model object initiated"
        )

        if model_config["cv_perform"]:
            assert (
                model_config["num_cross_validation"] is not None
                and model_config["param_grid_search"] is not None
            ), "cv_param_grid and cv_num should not be None."
            logging.info(
                f"Starting {model_config['num_cross_validation']}-fold cross validation"
                )

            model = GridSearchCV(
                estimator=model,
                param_grid=model_config["param_grid_search"],
                cv=model_config["num_cross_validation"],
            )

        model.fit(data_train, y_train)
        logging.info("Model training complete.")

        self._get_model_performance(
            model_code,
            model,
            (data_train, data_test, y_train, y_test),
            model_config["cv_perform"]
        )
        self._store_model(model_code, model, model_config["cv_perform"])

    @staticmethod
    def _init_model(model_type, random_state=42):
        """
        funciton to initiate model object

        input:
                model_type: str.
                random_state: int.

        output:
                model_object: model object
        """
        if model_type == "rfc":
            return RandomForestClassifier(random_state)

        elif model_type == "lrc":
            return LogisticRegression()

        else:
            assert False, f"model_type '{model_type}' is not valid model input."
            return None

    def _get_model_performance(
        self,
        model_code,
        model_object,
        data,
        cv_perform: bool,
    ):
        """
        function to record and store model performance plots

        inputs:
                model_type: str
                model_name: str
                model_object: trained model object
                data: tuple (data_train, data_test, y_train, y_test)
                cv_perform: bool. indicates whether cross validation was perform
        outputs:
                None
        """

        data_train = data[0]
        data_test = data[1]
        y_train = data[2]
        y_test = data[3]

        if cv_perform:
            y_train_preds = model_object.best_estimator_.predict(data_train)
            y_test_preds = model_object.best_estimator_.predict(data_test)

        else:
            y_train_preds = model_object.predict(data_train)
            y_test_preds = model_object.predict(data_test)

        # scores

        # save performance plots
        self.plot_classification_report(
            model_code, y_train, y_train_preds, y_test, y_test_preds
        )
        logging.info("Classification Report saved.")

        self.plot_aucroc_curve(model_code, model_object, data_test, y_test)
        logging.info("AUC-ROC Plot saved.")

        # model performance in explainability
        if model_code in ["rfc"]:
            self.plot_shap_explainer(
                model_code, model_object, data_test, cv_perform)
            logging.info("SHAP Plot saved.")

            self.plot_feature_importance(
                model_code, model_object, data_train, cv_perform)
            logging.info("Feature Importance Plot saved.")

    def _store_model(self, model_type, model_object, cv_perform):
        """
        function to store trained model

        intputs:
                model_type: str.
                model_object: trained model object
                cv_perform: bool. indicates whether cross validation was perform

        output:
                None
        """
        if cv_perform:
            joblib.dump(
                model_object.best_estimator_,
                self._model_path_format.format(model_type))
        else:
            joblib.dump(
                model_object,
                self._model_path_format.format(model_type))
        logging.info(
            f"Model successfully stored in '{self._model_path_format.format(model_type)}'"
        )

    @staticmethod
    def load_model(model_path: str):
        """
        funciton to load pretrained model

        input:
                model_path: file path to model file

        output:
                model: model obect
        """
        model = joblib.load(model_path)
        return model


if __name__ == "__main__":
    constant = read_json("constant.json")
    dataframe = import_data(constant["file_path"]["dataset"])

    dataframe = EncoderHelper.encode_churn_target_var(
        dataframe,
        constant["target_col"],
        constant["target_col_encode_val"],
        constant["target_var_name"],
    )
    dataframe = EncoderHelper.encode_cat_col(dataframe, constant["categorical_cols"], "Churn")

    # perform EDA
    eda_helper = EDAHelper(constant["file_path"]["image"])
    eda_helper.return_data_summary(dataframe)
    eda_helper.plot_hist(dataframe, "Churn")
    eda_helper.plot_hist(dataframe, "Customer_Age")
    eda_helper.plot_bar(dataframe, "Marital_Status")
    eda_helper.plot_dist(dataframe, "Total_Trans_Ct")
    eda_helper.plot_corr_heatmap(dataframe)

    # perform feature engineering
    data_train, data_test, y_train, y_test = perform_feature_engineering(
        dataframe, X_cols=constant["feature_cols"], y_col=constant["target_var_name"]
    )

    # Train model
    # Initialize model class
    model_wrapper = ChurnModel(
        model_dir=constant["file_path"]["model"],
        plot_dir=constant["file_path"]["image"],
    )

    # perform model training, store results, store model object in pkl
    for model_code in constant["model_config"].keys():
        model_wrapper.train(
            model_code,
            constant["model_config"][model_code],
            (data_train, data_test, y_train, y_test)
        )
