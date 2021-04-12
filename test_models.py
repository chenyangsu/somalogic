"""
Testing the model.
- Loads the trained model.
- Preprocesses the test set by standardizing protein levels based on training set mean and std.
    *** NOTE: Make sure that your dataset contains the columns named age_at_diagnosis, sex, ProcessTime, SampleGroup
- Predicts AUCs on test set using trained model.
"""

import os
import pickle
from utils import boolean
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from models import get_samples
import matplotlib.pyplot as plt
import scikitplot as skplt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic
DAT_DIR = os.path.join(ROOT_DIR, 'results', 'datasets')  # .../somalogic/results/datasets
TEST_DIR = os.path.join(DAT_DIR, 'test')  # .../somalogic/results/datasets/test
FINAL_MODEL_DIR = os.path.join(ROOT_DIR, 'results', 'models', 'final')


def split_x_y(df, outcome):
    """
    Splits the dataframe into x and y (outcome)
    :param df: dataframe to be split
    :param outcome: The outcome for setting the y labels
    :return: x dataframe for the samples, y dataframe for the ground truths
    """
    y = df[outcome]
    x = df.drop(outcome, axis=1)

    return x, y


def preprocess(df, prot_list, nat_log_transf):
    """
    Takes data and one-hot encodes dummy variables (e.g. sex becomes sex_F, sex_M) and drops
    one of the dummy variables (e,g, here sex_F is dropped) to prevent multicollinearity.
    Finally, it sorts the columns into order: age_at_diagnosis, sex_M, proteins
    :param df: The dataframe to be processed
    :param prot_list: list of protein names in the same order as in the training set
    :return: The preprocessed dataframe
    """
    #TODO: Add try except block to catch if ProcessTime or SampleGroup don't exist
    #TODO: check Edgar protein list is in our proteins
    #TODO: Remove NoneX proteins and retrain
    var = df[['age_at_diagnosis', 'sex', 'ProcessTime', 'SampleGroup']]

    prot = df[prot_list]  # use prot_list to order proteins test set in correct order
    var = pd.get_dummies(var)  # convert categorical variables to dummy variables

    print(var.head())  # use this to check column headers
    var.drop('sex_F', axis=1, inplace=True)  # drop one of dummy variables to prevent multicollinearity

    # If SampleGroup has more than 2 columns, drop one of them but I assume there will only be one column since your data
    # is from a single hospital
    # var.drop('SampleGroup_CHUM', axis=1, inplace=True)

    cols = {'SampleGroup.*': 'SampleGroup'}  # https://stackoverflow.com/a/46707076
    var.columns = var.columns.to_series().replace(cols, regex=True)

    assert False
    # rename whatever SampleGroup_* column you have to SampleGroup (i.e. change SampleGroup_JGH to the name of your SampleGroup_* column)
    var = var.rename({'SampleGroup_JGH': 'SampleGroup'}, axis='columns')  # rename SampleGroup_JGH to SampleGroup

    var = var[['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]  # resort columns to this order

    if nat_log_transf:
        prot = prot.apply(np.log, axis=1)  # equiv to df.sum(1)

    df = pd.concat([var, prot], axis=1)  # merge dataframes by column

    # df.fillna(df.mean(), inplace=True)  # fill na values with the mean
    return df

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--soma_data', type=str, default='normalized', choices=['normalized', 'unnormalized'],
                        help='The SomaLogic dataset to use')
    parser.add_argument('--nat_log_transf', type=boolean, default=True, choices=[True, False],
                        help='Whether to log transform the protein values')
    parser.add_argument('--standardize', type=boolean, default=True, choices=[True, False],
                        help='Whether to standardize the protein values to mean 0 and std 1')

    parser.add_argument('--data', type=str, default='infe',
                        choices=['infe', 'non_infe'], help='The dataset to use')
    parser.add_argument('--outcome', type=str, default='A2',
                        choices=['A2', 'A3', 'B2', 'C1'], help='The COVID severity outcome')
    parser.add_argument('--model_type', type=str, default='lasso',
                        choices=['lasso', 'elasticnet'], help='The model to perform the analysis')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    config = vars(parser.parse_args())

    data = config['data']
    outcome = config['outcome']
    nat_log_transf = config['nat_log_transf']
    standardize = config['standardize']
    model_type = config['model_type']

    soma_data = config['soma_data']

    file_path = TEST_DIR + '/' + 'test.csv'
    df = pd.read_csv(file_path, low_memory=False)
    X, y = split_x_y(df, outcome)  # split out outcome column
    # Use pickle to load the ordered list of proteins from training set
    X_choice = 'all_proteins'
    prot_list_file = f'{FINAL_MODEL_DIR}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_proteins.pkl'
    prot_list = pickle.load(open(prot_list_file, 'rb'))

    # df needs to have protein columns in exact same order as training dataset for StandardScaler() below to give accurate results
    df = preprocess(X, prot_list, nat_log_transf)  # age_at_diagnosis, sex_M, ProcessTime, SampleGroup, protein 1, ... , 5284 (same order of proteins as in training set)

    X_choices = ['baseline', 'all_proteins']

    for X_choice in X_choices:

        # load the model from disk
        final_model_results_path = f'{FINAL_MODEL_DIR}/{model_type}-{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}.pkl'
        model = pickle.load(open(final_model_results_path, 'rb'))

        X_test = get_samples(df=df, data=data, outcome=outcome, choice=X_choice, fdr=0.01)
        X_test_transf = X_test.copy()

        if X_choice == 'all_proteins':

            # load the StandardScaler and standardize on dataset (protein columns only)
            scaler_file = f'{FINAL_MODEL_DIR}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_scaler.pkl'
            scaler = pickle.load(open(scaler_file, 'rb'))

            features = X_test_transf[prot_list]
            features = scaler.transform(features.values)
            X_test_transf[prot_list] = features

            test_auc = roc_auc_score(y, model.predict_proba(X_test_transf)[:, 1])

        elif X_choice == 'baseline':  # don't standardize (since don't have proteins) and directly fit on X
            test_auc = roc_auc_score(y, model.predict_proba(X_test_transf)[:, 1])

        print(f"{X_choice} Test AUC score: {test_auc}")

        # result = loaded_model.score(X_test, Y_test)
        # print(result)

        # get predictions
        predictions = model.predict_proba(X_test_transf)

        # plot roc curve
        skplt.metrics.plot_roc(y, predictions)
        plt.show()