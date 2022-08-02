'''
This file is a set of functions performing tests and logging on churn library script

author: Wonyoung Seo
date: Sep 28, 2021
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_encode_churn_target_var(EncoderHelper, df):
    '''
    test encoding churn target variable
    '''

    try:
        df = EncoderHelper.encode_churn_target_var(
            df, "Attrition_Flag", "Existing Customer", "Churn")
        logging.info("Testing target variable encoding: SUCCESS")
        return df
    except KeyError as err:
        logging.error(
            "Testing target variable encoding: Target column does not exist in dataset")
        raise err


def test_encode_cat_col(EncoderHelper, df):
    '''
    test encoding categorical variables
    '''

    try:
        df = EncoderHelper.encode_cat_col(df,
                                          ["Gender",
                                           "Education_Level",
                                           "Marital_Status",
                                           "Income_Category",
                                           "Card_Category"],
                                          "Churn")
        logging.info("Testing categorical variable encoding: SUCCESS")
        return df
    except KeyError as err:
        logging.error(
            "Testing categorical variable encoding: encoding categorical variable failed")
        raise err


def test_eda_helper(EDAHelper, df):
    '''
    test perform eda function
    '''

    path = "./images/"

    try:

        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: It does not appear that the you "
                        "are correctly saving images to the eda folder.")
        raise err

    eda_helper = EDAHelper(path)
    eda_helper.return_data_summary(df)
    eda_helper.plot_hist(df, 'Churn')
    eda_helper.plot_hist(df, 'Customer_Age')
    eda_helper.plot_bar(df, 'Marital_Status')
    eda_helper.plot_dist(df, 'Total_Trans_Ct')
    eda_helper.plot_corr_heatmap(df)


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''

    feature_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]
    y_col = 'Churn'

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, X_cols=feature_cols, y_col=y_col)

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The four objects that should be returned were not.")
        raise err

    return X_train, X_test, y_train, y_test


def test_ChurnModel(model_class, data):
    '''
    test train_models
    '''

    model_path = "./models/"
    image_path = "./images/"

    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    try:
        dir_model_path = os.listdir(model_path)
        assert len(dir_model_path) > 0
    except AssertionError as err:
        logging.error("Testing ChurnModel: model_paths do not exist")
        raise err

    try:
        dir_image_path = os.listdir(image_path)
        assert len(dir_image_path) > 0
    except AssertionError as err:
        logging.error("Testing ChurnModel: image_paths do not exist")
        raise err

    model_wrapper = model_class(model_path, image_path)

    model_code = 'rfc'
    model_config = {
        "model_name": "Random Forest",
        "cv_perform": True,
        "param_grid_search": {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"]
        },
        "num_cross_validation": 2
    }
    model_wrapper.train(
        model_code,
        model_config,
        (X_train, X_test, y_train, y_test))


if __name__ == "__main__":
    DATA_FRAME = test_import(cls.import_data)
    DATA_FRAME = test_encode_churn_target_var(cls.EncoderHelper, DATA_FRAME)
    DATA_FRAME = test_encode_cat_col(cls.EncoderHelper, DATA_FRAME)
    test_eda_helper(cls.EDAHelper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA_FRAME)
    test_ChurnModel(cls.ChurnModel, (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST))
