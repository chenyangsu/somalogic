"""
Runs the L1 regularized logistic regression model on the dataset.
A few important points:
1. we use the SomaLogic Normalized dataset
2. We preprocess the protein levels by natural log transforming them
3. We standardize protein levels during Stratified 5 fold cross validation by running StandardScaler() on the 4 training folds and scaling the validation fold

Note: required feature columns of dataset: age_at_diagnosis, sex, ProcessTime, SampleGroup, 5284 proteins
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import seaborn as sns
import pickle
from utils import boolean

from plotting import plot_pca
from plotting import plot_nonzero_coefficients
from plotting import plot_auc
from plotting import plot_age_distribution
from plotting import plot_protein_level_distribution
from plotting import plot_correlation

sns.set_theme()
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic
DAT_DIR = os.path.join(ROOT_DIR, 'results', 'datasets')  # .../somalogic/results/datasets
# FDR_DAT_DIR = os.path.join(ROOT_DIR, 'results', 'all_proteins', 'age+sex+SampleGroup+ProcessTime+protlevel')
PROT_LIST = os.path.join(ROOT_DIR, 'data', 'Somalogic_list_QC1.txt')
SEED = 0


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


def preprocess(df, nat_log_transf):
    """
    Takes data and one-hot encodes dummy variables (e.g. sex becomes sex_F, sex_M) and drops
    one of the dummy variables (e,g, here sex_F is dropped) to prevent multicollinearity.
    Finally, it sorts the columns into order: age_at_diagnosis, sex_M, proteins
    :param df: The dataframe to be processed
    :return: The preprocessed dataframe
    """
    var = df[['age_at_diagnosis', 'sex', 'ProcessTime', 'SampleGroup']]

    # read in text file of proteins (4984 proteins)
    with open(PROT_LIST) as f:
        protein = f.readlines()
    prot_list = [x.strip() for x in protein]
    print(prot_list)
    print(len(prot_list))

    prot = df[prot_list] # proteins

    var = pd.get_dummies(var)  # convert categorical variables to dummy variables

    var.drop('sex_F', axis=1, inplace=True)  # drop one of dummy variables to prevent multicollinearity
    var.drop('SampleGroup_CHUM', axis=1, inplace=True)

    var = var.rename({'SampleGroup_JGH': 'SampleGroup'}, axis='columns')  # rename SampleGroup_JGH to SampleGroup
    # Note: 1 in SampleGroup column corresponds to JGH while 0 corresponds to CHUM

    var = var[['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]  # resort columns in this order

    if nat_log_transf:
        prot = prot.apply(np.log, axis=1)  # equiv to df.sum(1)

    df = pd.concat([var, prot], axis=1)  # merge dataframes by column

    # df.fillna(df.mean(), inplace=True)  # fill na values with the mean

    return df


def get_samples(df, data, outcome, choice, fdr=0.01):
    """
    Forms the dataset based on the choice chosen.
    ############################################
    baseline: age + sex
    all_proteins: age + sex + all 5284 proteins
    fdr_sig_proteins: age + sex + all proteins that survive fdr correction
    ABO: age + sex + ABO
    CRP: age + sex + CRP
    ABO_CRP: age + sex + ABO + CRP
    ############################################
    :param df: original dataset
    :param data: the dataset to use e.g. infe, non_infe
    :param outcome: The outcome of interest
    :param choice: str
    :param fdr: only for fdr_sig_proteins case.
    :return: data frame of the data set
    """

    if choice == 'baseline':
        df = df[['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]

    elif choice == 'all_proteins':
        df = df

    # elif choice == 'fdr_sig_proteins':
    #     # get list of proteins that survived fdr correction by reading in logistic regression results
    #     # from prior analysis
    #     file_path = FDR_DAT_DIR + '/' + '_'.join(['', data, outcome, 'LR',
    #                                               'age+sex+protlevel', 'Analysis=all_proteins.xlsx'])
    #     lr_results = pd.read_excel(file_path, engine='openpyxl')  # without engine, will not open
    #     lr_results_sig = lr_results[lr_results['FDRp'] < fdr]  # get subset surviving FDR
    #     fdr_sig_prot_names = list(lr_results_sig["Protein"])
    #
    #     df = pd.concat([df['age_at_diagnosis'], df['sex_M'], df[fdr_sig_prot_names]], axis=1)

    # elif choice == 'ABO':
    #     df = df[['age_at_diagnosis', 'sex_M', 'ABO.9253.52']]
    #
    # elif choice == 'CRP':
    #     df = df[['age_at_diagnosis', 'sex_M', 'CRP.4337.49']]
    #
    # elif choice == 'ABO_CRP':
    #     df = df[['age_at_diagnosis', 'sex_M', 'ABO.9253.52', 'CRP.4337.49']]

    else:
        raise NotImplementedError

    return df



def lasso(C=1.0, random_state=0):
    """
    L1 regularized logistic regression model
    :param C: Amount of regularization. Corresponds to the inverse of lambda. E.g. if lambda = 10, then C = 0.1
    :param random_state: Seeding for reproducibility.
    :return: Lasso model
    """
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=100, random_state=random_state)
    return model


def elasticnet(C=1.0, l1_ratio=0.0, random_state=0):
    """
    L1 and L2 regularized logistic regression model
    :param C: Amount of regularization. Corresponds to the inverse of lambda. E.g. if lambda = 10, then C = 0.1
    :param l1_ratio: Amount of l1 regularization. If l1_ratio = 0.1, then l2_ratio must be 0.9
    :param random_state: Seeding for reproducibility
    :return: elasticnet model
    """

    model = LogisticRegression(penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state)
    return model


def repeated_stratified_kfold_gridsearchcv(X,
                                           y,
                                           X_choice,
                                           standardize,
                                           model_type='lasso',
                                           hyperparams={},
                                           n_splits=5,
                                           n_repeats=10,
                                           random_state=0):
    """

    :param X: The design matrix
    :param y: The outcome
    :param standardize: Whether or not to standardize protein levels during cross validation. If True, standardizes
    protein levels using StandardScaler() on 4 training folds and then transforms the 1 validation fold on each
    cross-validation experiment
    :param X_choice: the dataset choice: "baseline" or "all_proteins"
    :param model_type: the model to use e.g. lasso
    :param hyperparams: The hyperparameters of the model
    :param n_splits: The number of training splits. Corresponds to "k" in K fold cross-validation
    :param n_repeats: The number of times to run cross validation. If n_splits=5, n_repeats=2, then 10 total experiments
    will be run (5 folds * 2 repeats).
    :param random_state: the seed for reproducibility
    :return: a dictionary containing the cross validation results
    """

    # Perform Grid Search on each parameter configuration and cross validation
    # if model_type == 'lasso':
    #     model = lasso(random_state)
    # elif model_type == 'elasticnet':
    #     model = elasticnet(random_state)
    # else:
    #     raise NotImplementedError

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    C = hyperparams['C']

    results = {'C': C, 'mean_train_score': [], 'mean_val_score': [], 'std_train_score': [], 'std_val_score': [], 'best_hyperparam': {}}

    if X_choice == 'all_proteins' and standardize == True:
        for i, c in enumerate(C):  # loop over hyperparameter
            train_aucs, val_aucs = [], []  # for storing aucs over all splits

            for j, (train_index, val_index) in enumerate(rskf.split(X, y)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                features_to_scale = list(X_train.columns)
                features_to_scale.remove('age_at_diagnosis')
                features_to_scale.remove('sex_M')
                features_to_scale.remove('ProcessTime')
                features_to_scale.remove('SampleGroup')  # drop since want to standardize proteins only

                X_train_transf = X_train.copy()
                features = X_train_transf[features_to_scale]
                scaler = StandardScaler().fit(features.values)  # fit scaler on X_train
                features = scaler.transform(features.values)
                X_train_transf[features_to_scale] = features
                # print(X_train_transf.head())

                X_val_transf = X_val.copy()
                features = X_val_transf[features_to_scale]
                features = scaler.transform(features.values)  # use scaler fitted on X_train to transform X_val
                X_val_transf[features_to_scale] = features
                # print(X_val_transf.head())

                clf = lasso(C=c)
                clf.fit(X_train_transf, y_train)

                train_auc = roc_auc_score(y_train, clf.predict_proba(X_train_transf)[:, 1])
                val_auc = roc_auc_score(y_val, clf.predict_proba(X_val_transf)[:, 1])

                train_aucs.append(train_auc)
                val_aucs.append(val_auc)
                print(f'Finished {X_choice} -- Hyperparameter {i+1}/{len(C)} (C={c}): Experiment {j+1}/{n_splits*n_repeats}')

            results['mean_train_score'].append(np.mean(train_aucs))
            results['mean_val_score'].append(np.mean(val_aucs))
            results['std_train_score'].append(np.std(train_aucs))
            results['std_val_score'].append(np.std(val_aucs))

    else:  # no standardization of protein levels during cross-validation ('baseline')
        for i, c in enumerate(C):  # loop over hyperparameter
            train_aucs, val_aucs = [], []  # for storing aucs over all splits

            for j, (train_index, val_index) in enumerate(rskf.split(X, y)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                clf = lasso(C=c)
                clf.fit(X_train, y_train)

                train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
                val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

                train_aucs.append(train_auc)
                val_aucs.append(val_auc)
                print(f'Finished {X_choice} -- Hyperparameter {i+1}/{len(C)} (C={c}): Experiment {j+1}/{n_splits*n_repeats}')

            results['mean_train_score'].append(np.mean(train_aucs))
            results['mean_val_score'].append(np.mean(val_aucs))
            results['std_train_score'].append(np.std(train_aucs))
            results['std_val_score'].append(np.std(val_aucs))

    idx_of_best_val_score = results['mean_val_score'].index(max(results['mean_val_score']))
    best_hyperparam_value = results['C'][idx_of_best_val_score]
    results['best_hyperparam']['C'] = best_hyperparam_value

    return results


def get_model_coefficients(clf, X):
    """
    Gets the trained model coefficients
    :param clf: The trained model
    :param X: The feature matrix
    :return: The coefficients in various lists
    """

    coef = list(clf.coef_[0])  # get coefficient values
    coef_names = X.columns.to_list()  # get coefficient variable names

    # get nonzero coefficients
    nonzero_coef_idx = np.nonzero(clf.coef_[0])[0]  # get the index of nonzero coefficients
    nonzero_coef = [coef[i] for i in nonzero_coef_idx]
    nonzero_coef_names = [coef_names[i] for i in nonzero_coef_idx]

    # sort the nonzero coefficients
    sorted_nonzero_coef = sorted(nonzero_coef, reverse=True)  # sort coefficients descending order
    sorted_nonzero_coef_names_idxs = [nonzero_coef.index(x) for x in
                                      sorted_nonzero_coef]  # get the index of sorted coefficients
    sorted_nonzero_coef_names = [nonzero_coef_names[i] for i in
                                 sorted_nonzero_coef_names_idxs]  # sort the coefficient names by index

    # sort the absolute value of the nonzero coefficients
    abs_nonzero_coef = [abs(i) for i in nonzero_coef]  # get absolute value of nonzero coefficients
    abs_sorted_nonzero_coef = sorted(abs_nonzero_coef, reverse=True)  # sort coefficients descending order
    abs_sorted_nonzero_coef_names_idxs = [abs_nonzero_coef.index(x) for x in
                                      abs_sorted_nonzero_coef]  # get the index of sorted coefficients
    abs_sorted_nonzero_coef_names = [nonzero_coef_names[i] for i in abs_sorted_nonzero_coef_names_idxs]

    print(f'{X_choice}')
    print(nonzero_coef)
    print(nonzero_coef_names)
    print(len(nonzero_coef))

    print(sorted_nonzero_coef)
    print(sorted_nonzero_coef_names)

    print(abs_sorted_nonzero_coef)
    print(abs_sorted_nonzero_coef_names)

    return coef, coef_names, nonzero_coef, nonzero_coef_names, \
           sorted_nonzero_coef, sorted_nonzero_coef_names, abs_sorted_nonzero_coef, abs_sorted_nonzero_coef_names


def get_hyperparams(model_type):
    """
    Get hyperparameters to search over for specific model
    :param model_type: the model of interest
    :return: a dictionary containing the hyperparameters to search over
    """
    C_range = [10 ** (x / 4) for x in range(-8, 9)]  # C = 1/lambda
    #print(f'C range: {C_range}')

    l = [math.log10(1 / c) for c in C_range]  # lambda
    # print(f'Lambda range: {l}')

    if model_type == 'lasso':
        hyperparams = {'C': C_range}

    elif model_type == 'elasticnet':
        l1_ratio = [x * 0.1 for x in range(0, 11)]
        hyperparams = {'C': C_range, 'l1_ratio': l1_ratio}

    return hyperparams


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
    parser.add_argument('--params_search', type=boolean, default=True,
                        choices=[True, False], help='Whether or not to perform a hyperparameter seach of the model'
                                                    'specified in --model_type. If False, directly loads '
                                                    'model results which are presumed to be saved already and runs the '
                                                    'model on the entire training data.')
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
    if soma_data == 'normalized':
        file_path = DAT_DIR + '/' + 'infe_417-soma_data=normalized-nat_log_tranf=FALSE-standardize=FALSE-remove_outliers=FALSE.csv'
    elif soma_data == 'unnormalized':
        file_path = DAT_DIR + '/' + 'infe_417-soma_data=unnormalized-nat_log_tranf=FALSE-standardize=FALSE-remove_outliers=FALSE.csv'

    df = pd.read_csv(file_path, low_memory=False)

    X, y = split_x_y(df, outcome)
    df = preprocess(X, nat_log_transf)  # age_at_diagnosis, sex_M, ProcessTime, SampleGroup, protein 1, ... , 5284

    # Look at data
    # plot_pca(df=df, y=y, data=data, outcome=outcome, cluster_by='samples', num_components=20)
    # plot_age_distribution(df=df, y=y, data=data, outcome=outcome)
    # plot_protein_level_distribution(df=df, y=y, data=data, outcome=outcome, prot_list=df.columns.tolist()[2:7])  # plot distribution of first 5 proteins

    X_choices = ['baseline', 'all_proteins']

    hyperparams = get_hyperparams(model_type=model_type)
    # print(hyperparams)

    # colors = ['#d53e4f', 'lightcoral', 'blue', 'purple', 'lime', '#fee08b', '#fc8d59']
    colors = ['#d53e4f', 'blue']
    model_results = {}

    model_dir = os.path.join(ROOT_DIR, 'results', 'models')
    model_results_path = f'{model_dir}/{model_type}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_cv_results.pkl'

    if config['params_search']:  # run hyperparam search and save model results

        os.makedirs(model_dir, exist_ok=True)

        for X_choice in X_choices:
            X = get_samples(df=df, data=data, outcome=outcome, choice=X_choice, fdr=0.01)

            cv_results = repeated_stratified_kfold_gridsearchcv(X,
                                                                y,
                                                                X_choice,
                                                                standardize=standardize,
                                                                hyperparams=hyperparams,
                                                                model_type=model_type,
                                                                n_splits=5,
                                                                n_repeats=10,
                                                                random_state=SEED)
            model_results[X_choice] = cv_results

        # save model parameters
        with open(model_results_path, 'wb') as fp:
            pickle.dump(model_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'...Model results saved at {model_results_path}')

    # load model parameters
    with open(model_results_path, 'rb') as fp:
        model_results = pickle.load(fp)
    print(model_results.keys())
    print(model_results)

    plot_auc(model_type=model_type,
                    data=data,
                    outcome=outcome,
                    hyperparams=hyperparams,
                    model_results=model_results,
                    colors=colors)

    # use best hyperparameter to train on entire dataset

    # create directory for saving final model
    final_model_dir = os.path.join(ROOT_DIR, 'results', 'models', 'final')
    os.makedirs(final_model_dir, exist_ok=True)

    complete_summary = {}  # for storing the mean and std of each protein
    for i, X_choice in enumerate(X_choices):

        C = model_results[X_choice]['best_hyperparam']['C']
        clf = lasso(C=C, random_state=SEED)
        X = get_samples(df=df, data=data, outcome=outcome, choice=X_choice, fdr=0.01)

        # plot CV ROC
        #####################################

        # rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
        #
        # fig, ax = plt.subplots(figsize=(10, 8))
        #
        # mean_fpr = np.linspace(0, 1, 100)
        # tprs = []
        # aucs = []
        # cf_matrix = np.zeros((2, 2))
        # for j, (train_index, test_index) in enumerate(rskf.split(X, y)):
        #     # print("TRAIN:", train_index, "TEST:", test_index)
        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #
        #     lr = lasso(C=C, random_state=SEED)
        #
        #     lr.fit(X_train, y_train)
        #
        #     viz = plot_roc_curve(lr, X_test, y_test,
        #                          name='ROC fold {}'.format(j),
        #                          alpha=0.3, lw=1, ax=ax)
        #     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        #     interp_tpr[0] = 0.0
        #     tprs.append(interp_tpr)
        #     aucs.append(viz.roc_auc)
        # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        #         label='Chance', alpha=.8)
        #
        # mean_tpr = np.mean(tprs, axis=0)
        # mean_tpr[-1] = 1.0
        # mean_auc = auc(mean_fpr, mean_tpr)
        # std_auc = np.std(aucs)
        # ax.plot(mean_fpr, mean_tpr, color=colors[i],
        #         label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        #         lw=2, alpha=.8)
        #
        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        #
        # ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        #        title=f"{data} {outcome} - {X_choice} - Receiver operating characteristic")
        #
        # ax.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, prop={'size': 6})
        # plt.show()

        ## plotting other stats
        #     y_pred = lr.predict(X_test)  # get predictions
        #
        #     # Get the confusion matrix
        #     cf_matrix = confusion_matrix(y_test, y_pred)
        #     print(cf_matrix)


        #
        #     group_names = ['TN', 'FP', 'FN', 'TP']
        #     group_counts = ['{0:0.0f}'.format(value) for value in
        #                     cf_matrix.flatten()]
        #     group_percentages = ['{0:.2%}'.format(value) for value in
        #                          cf_matrix.flatten()/np.sum(cf_matrix)]
        #     labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
        #               zip(group_names,group_counts,group_percentages)]
        #     labels = np.asarray(labels).reshape(2,2)
        #     sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',
        #                 xticklabels=['Controls', 'Cases'],
        #                yticklabels=['Controls', 'Cases'])
        #     plt.xlabel('Predicted Values')
        #     plt.ylabel('True Values')
        #

        # assert False

        ######################################

        if X_choice == 'all_proteins':  # standardize protein levels (skipped if X_choice='baseline' dataset)
            features_to_scale = list(X.columns)
            features_to_scale.remove('age_at_diagnosis')
            features_to_scale.remove('sex_M')
            features_to_scale.remove('ProcessTime')
            features_to_scale.remove('SampleGroup')  # drop since want to standardize proteins only

            print(features_to_scale)  # features_to_scale should be a list of all 5284 protein names

            for feature in features_to_scale:
                summary = {}
                summary['mean'] = X[feature].mean(axis=0)
                summary['std'] = X[feature].std(axis=0, ddof=0)  # ddof=0 (default = 1) to be consistent with StandardScaler()
                complete_summary[feature] = summary  # store dictionary containing mean and std of protein inside dictionary

            # See what model coefficient values are if final training done on standardized vs. nonstandardized data
            X_transf = X.copy()
            features = X_transf[features_to_scale]
            scaler = StandardScaler().fit(features.values)  # fit scaler on X_train; uses same mean and std as complete_summary
            features = scaler.transform(features.values)
            X_transf[features_to_scale] = features

            # print(X_transf.head())  # check that proteins in X (for 'all_proteins') is standardized properly
            clf.fit(X_transf, y)

            # save the scaler dictionary
            scaler_file = f'{final_model_dir}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_scaler.pkl'
            pickle.dump(complete_summary, open(scaler_file, 'wb'))

        elif X_choice == 'baseline':  # don't standardize (since don't have proteins) and directly fit on X
            # print(X.head())
            clf.fit(X, y)

        else:
            raise NotImplementedError

        # save final model
        final_model_results_path = f'{final_model_dir}/{model_type}-{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}.pkl'
        pickle.dump(clf, open(final_model_results_path, 'wb'))

        # plot nonzero coefficients of model
        coef, coef_names, nonzero_coef, nonzero_coef_names, \
        sorted_nonzero_coef, sorted_nonzero_coef_names, abs_sorted_nonzero_coef, abs_sorted_nonzero_coef_names = get_model_coefficients(clf=clf, X=X)

        # plot nonzero coefficient values to determine the effect sizes
        plot_nonzero_coefficients(type='nonzero_coef', x_val=nonzero_coef, y_val=nonzero_coef_names,
                                  data=data, outcome=outcome, model_type=model_type, X_choice=X_choice, color=colors[i])

        plot_nonzero_coefficients(type='sorted_nonzero_coef', x_val=sorted_nonzero_coef, y_val=sorted_nonzero_coef_names,
                                  data=data, outcome=outcome, model_type=model_type, X_choice=X_choice, color=colors[i])

        plot_nonzero_coefficients(type='abs_sorted_nonzero_coef', x_val=abs_sorted_nonzero_coef, y_val=abs_sorted_nonzero_coef_names,
                                  data=data, outcome=outcome, model_type=model_type, X_choice=X_choice, color=colors[i])

        # plot spearman correlation of coefficients
        nonzero_prot_list = [protein for protein in nonzero_coef_names if protein not in ['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]  # keep only proteins

        if X_choice == 'all_proteins':
            plot_correlation(df=X, y=y, data=data, outcome=outcome, prot_list=nonzero_prot_list)
            # plot_correlation(df=X, y=y, data=data, outcome=outcome, prot_list=X.columns.to_list()[4:54])
