'''
This file is a set of functions for customer churn predction project

Auther: Wonyoung Seo
Date: Sep 27, 2021
'''

import os
import logging
from typing import List
import joblib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
# import shap


logging.basicConfig(
    level = logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)


def read_json(file_path: str) -> dict:
    '''
    function to read json file.
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)
    return data


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv

    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


class encoder_helper:

    def __init__(self):
        pass

    def encode_churn_target_var(self, df: pd.DataFrame, target_col: str, target_val: str, response: str) -> pd.DataFrame:
        '''
        function to convert churn column into target variable

        inputs:
                df: pandas dataframe.
                target_var_col: string.
                reponse: string.

        output:
                df: pandas dataframe
        '''
        assert target_val in df[target_col].unique(), "Value '{}' does not exist in target column '{}'".format(target_val, target_col)
        df[response] = df[target_col].apply(lambda val: 0 if val == target_val else 1)
        return df



    def encode_cat_col(self, df, category_lst, response) -> pd.DataFrame:
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''

        for cat in category_lst:

            encoded_lst = []
            group = df.groupby(cat).mean()[response]
            for val in df[cat]:
                encoded_lst.append(group.loc[val])
            df['{}_{}'.format(cat, response)] = encoded_lst

        return df


def perform_feature_engineering(df, X_cols: List[str], y_col: str, test_size: float = 0.3):
    '''
    function to prepare data for model training and split them into train, test dataset

    input:
              df: pandas dataframe
              X_cols: list of columns. contains independent variables/features
              y_cols: string. target variable
              test_size: ratio of test set out of total dataset. default 0.3

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    y = df[y_col]
    X = df[X_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test





class PlotGenerator:
    '''
    Wrapper class for generating plots
    '''

    def __init__(self, plot_file_path:str):
        self.plot_path = plot_file_path

    def _save_plot(self, plt_object, prefix: str, plot_type: str):
        '''
        function to save the generated plot

        input:
                plt_object:
                prefix: str
                plot_type: str

        output:
                None
        '''
        file_name = '{}_{}.png'.format(model_type, plot_type)
        file_path = os.path.join(self.plot_path, file_name)
        plt_object.savefig(file_path, bbox_inches='tight')
        plt_object.close()
        logging.info("Plot saved '{}'".format(file_path))

    def plot_hist(self, df, col_name):
        '''
        creates and store histogram plot on given column
        '''
        plt.figure()
        ax = plt.gcf()
        df[col_name].hist()
        self._save_plot(plt, col_name, plot_type='histogram')

    def plot_bar(self, df, col_name):
        '''
        creates and store bar plot on given column
        '''
        plt.figure()
        ax = plt.gcf()
        df[col_name].value_counts('normalize').plot(kind='bar')
        self._save_plot(plt, col_name, plot_type='bar')

    def plot_dist(self, df, col_name):
        '''
        creates and store distribution plot on given column
        '''
        plt.figure()
        ax = plt.gcf()
        sns.distplot(df[col_name]);
        self._save_plot(plt, col_name, plot_type='distribution')

    def plot_corr_heatmap(self, df, color_map='Dark2_r'):
        '''
        creates and store heatmap on given data
        '''
        plt.figure()
        ax = plt.gcf()
        sns.heatmap(df.corr(), annot=False, cmap=color_map, linewidths=2)
        self._save_plot(plt, 'corr', plot_type='heatmap')


    def plot_classification_report(self, model_name, y_train, y_train_preds, y_test, y_test_preds):
        '''
        creates classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds: training predictions
                y_test_preds: test predictions

        output:
                 None
        '''

        plt.rc('figure', figsize=(7, 7))

        plt.text(0.01, 1.25, str('{} Train'.format(model_name)), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!

        plt.text(0.01, 0.6, str('{} Test'.format(model_name)), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')

        self._save_plot(plt, model_name, 'classification_report')

    def plot_aucroc_curve(self, model_name, model_object, X_test, y_test):
        '''
        creates and stores the AUC-ROC curve of the model

        inputs:
            model_name: str
            model_object: trained model object
            X_test: test dataset
            y_test: test target variable

        output:
            None
        '''
        plt.figure()
        ax = plt.gcf()
        plot = plot_roc_curve(model_object, X_test, y_test, alpha=0.8)
        self._save_plot(plt, model_name, 'roc_curve')

    def plot_shap_explainer(self, model_name, model_object, X_test, cv_perform):
        '''
        creates and stores the shap values per feature

        inputs:
            model_name: str
            model_object: trained model object
            X_test: test dataset
            cv_perform: bool. indicates whether cross validation was perform

        output:
            None
        '''
        plt.figure(figsize=(20, 5))
        ax = plt.gcf()
        if cv_perform:
            explainer = shap.TreeExplainer(model_object.best_estimator_)
        else:
            explainer = shap.TreeExplainer(model_object)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, axis_color='#000000')
        self._save_plot(plt, model_name, 'shap')

    def plot_feature_importance(self, model_name, model_object, X_train, cv_perform):
        '''
        creates and stores the feature importances

        inputs:
            model_name: str
            model_object: trained model object
            X_train: train dataset
            cv_perform: bool. indicates whether cross validation was perform

        output:
            None
        '''

        # Calculate feature importance
        if cv_perform:
            importance = model_object.best_estimator_.feature_importances_
        else:
            importance = model_object.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importance)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_train.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_train.shape[1]), importance[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_train.shape[1]), names, rotation=90)

        self._save_plot(plt, model_name, 'feature_importance')


class EDA_Helper(PlotGenerator):
    '''
    perform eda on df and save figures to images folder
    '''

    @staticmethod
    def return_data_summary(df: pd.DataFrame):
        '''
        function to return statistics of the data

        input:
                df: pandas DataFrame
        ouput:
                None
        '''
        print("Data Shape:")
        df.info()
        print('\n')
        print("Statistics per column:")
        print(df.describe())




class ChurnModel(PlotGenerator):
    '''
    Model object for churn model.
    '''
    def __init__(self, model_dir: str='./models/', plot_dir:str='./image/', random_state:int=42):
        super().__init__(plot_dir)
        model_filename_format = '{}_model.pkl'
        self._model_path_format = os.path.join(model_dir, model_filename_format)
        self.random_state = random_state


    def train(self,
              model_type: str,
              model_name: str,
              X_train,
              X_test,
              y_train,
              y_test,
              cv_perform: bool = False,
              cv_param_grid: dict = None,
              cv_num: int = None):
        '''
        train and store model and model performance record

        input:
                model_type: str
                model_name: str
                X_train: train dataset
                X_test: test dataset
                y_train: train target set
                y_test: test target set
                cv_perform: bool. indicates whether to perform cross validation. default=False
                cv_param_grid: dict. dictionary of grid_search parameters. default=None
                cv_num: int. number of cross validation to perform. default=None

        output:
                None
        '''

        model_name = model_name_dict[model_type]

        model = self._init_model(model_type, self.random_state)
        logging.info('{} model object initiated'.format(model_name))

        if cv_perform:
            assert (cv_num is not None and cv_param_grid is not None), "cv_param_grid and cv_num should not be None."
            logging.info("Starting {}-fold cross validation".format(cv_num))
            model = GridSearchCV(estimator=model, param_grid=cv_param_grid, cv=cv_num)


        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        self._get_model_performance(model_type, model_name, model, X_train, X_test, y_train, y_test, cv_perform)
        self._store_model(model_type, model, cv_perform)

    def _get_model_performance(self, model_type, model_name, model_object, X_train, X_test, y_train, y_test, cv_perform: bool):
        '''
        function to record and store model performance plots

        inputs:
                model_type: str
                model_name: str
                model_object: trained model object
                X_train: train dataset
                X_test: test dataset
                y_train: train target set
                y_test: test target set
                cv_perform: bool. indicates whether cross validation was perform
        outputs:
                None
        '''

        if cv_perform:
            y_train_preds = model_object.best_estimator_.predict(X_train)
            y_test_preds = model_object.best_estimator_.predict(X_test)

        else:
            y_train_preds = model_object.predict(X_train)
            y_test_preds = model_object.predict(X_test)


        # scores

        # save performance plots
        self.plot_classification_report(model_name, y_train, y_train_preds, y_test, y_test_preds)
        logging.info("Classification Report saved.")

        self.plot_aucroc_curve(model_name, model_object, X_test, y_test)
        logging.info("AUC-ROC Plot saved.")

        self.plot_shap_explainer(model_name, model_object, X_test, cv_perform)
        logging.info("SHAP Plot saved.")

        if model_type in ['rfc']:
            self.plot_feature_importance(model_name, model_object, X_train, cv_perform)
            logging.info("Feature Importance Plot saved.")


    @staticmethod
    def _init_model(model_type, random_state=42):
        '''
        funciton to initiate model object

        input:
                model_type: str.
                random_state: int.

        output:
                model_object: model object
        '''
        if model_type == 'rfc':
            return RandomForestClassifier(random_state)
        elif model_type == 'lrc':
            return LogisticRegression()
        else:
            assert False, "model_type '{}' is not valid model input.".format(model_type)



    def _store_model(self, model_type, model_object, cv_perform):
        '''
        function to store trained model

        intputs:
                model_type: str.
                model_object: trained model object
                cv_perform: bool. indicates whether cross validation was perform

        output:
                None
        '''
        if cv_perform:
            joblib.dump(model_object.best_estimator_, self._model_path_format.format(model_type))
        else:
            joblib.dump(model_object, self._model_path_format.format(model_type))
        logging.info("Model successfully stored in '{}'".format(self._model_path_format.format(model_type)))


    def load_model(self, model_path: str):
        '''
        funciton to load pretrained model

        input:
                model_path: file path to model file

        output:
                model: model obect
        '''
        model = joblib.load(model_path)
        return model



if __name__ == "__main__":
    constant = read_json('constant.json')
    df = import_data(constant['file_path']['dataset'])

    encoder = encoder_helper()
    df = encoder.encode_churn_target_var(df,
                                         constant['target_col'],
                                         constant['target_col_encode_val'],
                                         constant['target_var_name'])
    df = encoder.encode_cat_col(df, constant['categorical_cols'], 'Churn')

    X_train, X_test, y_train, y_test = perform_feature_engineering(df,
                                                                   X_cols=constant['feature_cols'],
                                                                   y_col=constant['target_var_name'])


    # Train model
    ## Initialize model class
    model_wrapper = ChurnModel()

    ## perform model training, store results, store model object in pkl
    for model_type in constant['model_type_name_mapping'].keys():
        model_wrapper.train(model_type,
                            constant['model_type_name_mapping'][model_type],
                            X_train, X_test, y_train, y_test,
                            cv_perform=True,
                            cv_param_grid=constant['param_grid_search'],
                            cv_num=constant['num_cross_validation'])

