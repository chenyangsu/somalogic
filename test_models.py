"""
Testing the model.
- Loads the trained model.
- Preprocesses the test set by standardizing protein levels based on training set mean and std.
    *** NOTE: Make sure that your dataset contains the columns named age_at_diagnosis, sex, ProcessTime, SampleGroup
    - For any features in the training set that were not present in the test set, we added the feature into the test
    set and set it to the mean value from the training set.
        - For ProcessTime: we add a feature column to the test set using the mean value from the training set
        - For proteins: we add a single feature column to the test set using the mean value from the training set. Since
        protein levels are then standardized, the final protein level passed into the model is 0 ( (mean - mean)/std = 0 )
- Predicts AUCs on test set using trained model.
"""

import os
import pickle
from utils import boolean
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
import scikitplot as skplt
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic
DAT_DIR = os.path.join(ROOT_DIR, 'results', 'datasets')  # .../somalogic/results/datasets
TEST_DIR = os.path.join(DAT_DIR, 'test')  # .../somalogic/results/datasets/test
FINAL_MODEL_DIR = os.path.join(ROOT_DIR, 'results', 'models', 'final')
TEST_PROT_LIST = os.path.join(ROOT_DIR, 'data', 'mssm_protein_list.csv')


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


def get_samples(df, choice):
    cols = df.columns.tolist()
    clinical_col = [col for col in cols if col in ['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]

    if choice == 'baseline':
        df = df[clinical_col]

    elif choice == 'all_proteins':
        df = df

    return df


def preprocess(df, scaler_dict, train_features, nat_log_transf):
    """
    Takes data and one-hot encodes dummy variables (e.g. sex becomes sex_F, sex_M) and drops
    one of the dummy variables (e,g, here sex_F is dropped) to prevent multicollinearity.
    Finally, it sorts the columns into order: age_at_diagnosis, sex_M, proteins
    :param df: The dataframe to be processed
    :param scaler_dict: The dictionary containing mean and std of each feature from training set
    :return: The preprocessed dataframe
    """
    clinical_features = [f for f in train_features if
                 f in ['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]

    # leave only protein names
    prot_features = [f for f in train_features if
                    f not in ['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]

    ###### clinical features ######
    # try except block to catch if ProcessTime doesn't exist
    try:
        var = df[['age_at_diagnosis', 'sex', 'ProcessTime', 'SampleGroup']]
    except:
        v = df[['age_at_diagnosis', 'sex', 'SampleGroup']]
        mean_process_time = scaler_dict['ProcessTime']['mean']  # get mean value from training ProcessTime feature
        var = v.copy()
        var['ProcessTime'] = mean_process_time

    var = pd.get_dummies(var)  # convert categorical variables to dummy variables
    var.drop('sex_F', axis=1, inplace=True)  # drop one of dummy variables to prevent multicollinearity
    # If SampleGroup has more than 2 columns, drop one of them but I assume there will only be one column since your data
    # is from a single hospital

    # rename to SampleGroup
    cols = {'SampleGroup.*': 'SampleGroup'}  # https://stackoverflow.com/a/46707076
    var.columns = var.columns.to_series().replace(cols, regex=True)  # replace SampleGroup_... to SampleGroup

    clinical_features = [i if i != 'sex' else "sex_M" for i in clinical_features]  # change sex to sex_M

    var = var[clinical_features]  # reorder columns

    ###### proteins ######
    test_sum_stats = pd.read_csv(TEST_PROT_LIST, low_memory=False)
    test_prot_list = test_sum_stats['c'].tolist()  # Mt. Sinai list of proteins

    prot = df[test_prot_list]
    if nat_log_transf:  # first natural log the protein levels
        prot = prot.apply(np.log, axis=1)  # equiv to df.sum(1)

    # list of proteins in model that isn't in Mt. Sinai proteins
    extra_prot = [coef for coef in prot_features if coef not in test_prot_list]

    for p in extra_prot:  # Add proteins in model that aren't in test set to the test set and use the mean value from training
        prot_val = scaler_dict[p]['mean']  # get mean value from training
        prot[p] = prot_val  # add protein to dataframe

    prot = prot[prot_features]  # reorder proteins

    df = pd.concat([var, prot], axis=1)  # merge dataframes by column

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

    X_choice = 'all_proteins'

    # load the scaler dictionary and standardize on dataset (protein columns only)
    scaler_file = f'{FINAL_MODEL_DIR}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_scaler.pkl'
    scaler_dict = pickle.load(open(scaler_file, 'rb'))

    model_coef_file = f'{FINAL_MODEL_DIR}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_coef.pkl'
    model_coef = pickle.load(open(model_coef_file, 'rb'))

    coefficients = list(model_coef.keys())

    train_features_file = f'{FINAL_MODEL_DIR}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_train_features.pkl'
    with open(train_features_file, "rb") as fp:  # Unpickling
        train_features = pickle.load(fp)

        #print((list(model_coef.keys())))

    df = preprocess(X, scaler_dict, train_features, nat_log_transf)  # age_at_diagnosis, sex_M, ProcessTime, SampleGroup, protein 1, ... , 5284 (same order of proteins as in training set)

    X_choices = ['baseline', 'all_proteins']

    for X_choice in X_choices:

        # load the model from disk
        final_model_results_path = f'{FINAL_MODEL_DIR}/{model_type}-{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}.pkl'
        model = pickle.load(open(final_model_results_path, 'rb'))

        X_test = get_samples(df=df, choice=X_choice)

        X_test_transf = X_test.copy()

        if X_choice == 'all_proteins':
            features = [f for f in train_features if
                             f not in ['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]
            for p in features:  # for each protein,  standardize based on mean and std from training set
                X_test_transf[p] = (X_test_transf[p] - scaler_dict[p]['mean']) / scaler_dict[p]['std']

            print(f"Test set for {X_choice}:\n{X_test_transf.head()}\n")
            test_auc = roc_auc_score(y, model.predict_proba(X_test_transf)[:, 1])

        elif X_choice == 'baseline':  # don't standardize (since don't have proteins) and directly fit on X

            print(f"Test set for {X_choice}\n{X_test_transf.head()}\n")
            test_auc = roc_auc_score(y, model.predict_proba(X_test_transf)[:, 1])

        print(f"({X_choice}) Test AUC score: {test_auc}\n")
        print('#'*100, end='\n')
        # result = loaded_model.score(X_test, Y_test)
        # print(result)

        # get prediction probabilities for calculating AUC
        y_pred_proba = model.predict_proba(X_test_transf)[:, 1]  # cases

        fpr, tpr, _ = metrics.roc_curve(y, y_pred_proba)
        auc = metrics.roc_auc_score(y, y_pred_proba)

        plt.plot(fpr, tpr, label="Test Set, auc=" + "{:.3f}".format(auc))  # plot roc curve
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)  # plot diagonal line
        plt.legend(loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        fig_path = os.path.join(TEST_DIR, f'auc_curve_{X_choice}')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.show()

        y_pred = model.predict(X_test_transf)  # get predictions
        cf_matrix = confusion_matrix(y, y_pred)
        print(cf_matrix)

        # print(np.array(y))
        dict = {'y': y,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred,
                'fpr': fpr,
                'tpr': tpr}

        file_name = os.path.join(TEST_DIR, f'test_results_{X_choice}.pkl')
        with open(file_name, 'wb') as fp:
            pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # # plot roc curve
        # skplt.metrics.plot_roc(y, predictions)
        # plt.show()