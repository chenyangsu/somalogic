"""
Runs the L1 regularized logistic regression model on the dataset for sensitivity analysis
A few important points:
1. we use the SomaLogic Normalized dataset
2. We preprocess the protein levels by natural log transforming them
3. We standardize protein levels during Stratified 5 fold cross validation by running StandardScaler() on the 4 training folds and scaling the validation fold

NOTES: ------------------------------------------------------
Baseline model: age_at_diagnosis, sex, ProcessTime, SampleGroup, 6 clinical comorbidities
Protein model: age_at_diagnosis, sex, ProcessTime, SampleGroup + 6 clinical comorbidities + 4984 proteins

Here, we don't use smoking as a feature and thus can use the entire sample size of 417 for training

Furthermore, for the 6 clinical features encoding is as follows:
- com_diabetes: Diabetes - "0 No, 1 Yes, -1 Don't know"
- com_chronic_pulm: Chronic obstructive pulmonary disease (COPD) - "0 No, 1 Yes, -1 Don't know"
- com_chronic_kidney: Chronic kidney disease - "0 No, 1 Yes, -1 Don't know"
- com_heart_failure: Congestive heart failure - "0 No, 1 Yes, -1 Don't know"
- com_hypertension: Hypertension - "0 No, 1 Yes, -1 Don't know"
- com_liver: Liver Disease - "0 No, 1 Yes, -1 Don't know"

For the 6 clinical features, we set anything other than 1 to a 0 to form a binary variable
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
from plotting import plot_auc
from plotting import plot_age_distribution
from plotting import plot_protein_level_distribution

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
from tableone import TableOne

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic
DAT_DIR = os.path.join(ROOT_DIR, 'results', 'datasets')  # .../somalogic/results/datasets
# FDR_DAT_DIR = os.path.join(ROOT_DIR, 'results', 'all_proteins', 'age+sex+SampleGroup+ProcessTime+protlevel')
PROT_LIST = os.path.join(ROOT_DIR, 'data', 'Somalogic_list_QC1.txt')
SEED = 0

CLINICAL_FILE = os.path.join(ROOT_DIR, 'data', 'basic_JGH_CHUM_20210413.csv')
DATA_FILE = os.path.join(ROOT_DIR, 'results', 'datasets',
                         'infe_417-soma_data=normalized-nat_log_tranf=FALSE-standardize=FALSE-remove_outliers=FALSE.csv')
NON_PROT_FEAT = ['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup',
                 'com_diabetes',
                 'com_chronic_pulm',
                 'com_chronic_kidney',
                 'com_heart_failure',
                 'com_hypertension',
                 'com_liver']

PLOTS_DIR = os.path.join(ROOT_DIR, 'results', 'plots')

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

    prot = df[prot_list]  # proteins

    var = pd.get_dummies(var)  # convert categorical variables to dummy variables

    var.drop('sex_F', axis=1, inplace=True)  # drop one of dummy variables to prevent multicollinearity
    var.drop('SampleGroup_CHUM', axis=1, inplace=True)

    var = var.rename({'SampleGroup_JGH': 'SampleGroup'}, axis='columns')  # rename SampleGroup_JGH to SampleGroup
    # Note: 1 in SampleGroup column corresponds to JGH while 0 corresponds to CHUM

    var['anonymized_patient_id'] = df['anonymized_patient_id']
    var = var[['anonymized_patient_id', 'age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]  # resort columns in this order

    if nat_log_transf:
        prot = prot.apply(np.log, axis=1)  # equiv to df.sum(1)

    df = pd.concat([var, prot], axis=1)  # merge dataframes by column

    # df.fillna(df.mean(), inplace=True)  # fill na values with the mean

    clinical = pd.read_csv(CLINICAL_FILE)

    columns = ['anonymized_patient_id',
               'com_diabetes',
               'com_chronic_pulm',
               'com_chronic_kidney',
               'com_heart_failure',
               'com_hypertension',
               'com_liver']

    clinical = clinical[columns]

    mytable = TableOne(clinical, columns=columns[1:])
    print(mytable.tabulate(tablefmt="fancy_grid"))

    for feature in clinical.columns.tolist()[1:]:  # excluding anonymized_patient_id
        clinical[feature].values[clinical[feature].values != 1] = 0

    # pd.read_csv converts all values to float due to having NaNs so values back to int
    for feature in clinical.columns.tolist()[1:]:  # excluding anonymized_patient_id
        clinical[feature] = clinical[feature].astype(int)

    clinical = clinical.applymap(str)  # convert all int64 dtypes to object dtypes so can do dummy encoding

    # dummy encoding for smoking (other comorbidities are 0 and 1 already)
    df = pd.merge(df, clinical, how='inner', on='anonymized_patient_id')  # GOTCHA: merge (df, clinical) NOT (clinical, df) which will change order of rows to clinical dataset
    print(df.head())
    print(df.shape)
    df = df.drop(columns=['anonymized_patient_id'])

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
        df = df[NON_PROT_FEAT]
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



def lasso(C=1.0, random_state=0, tol=0.0001):
    """
    L1 regularized logistic regression model
    :param C: Amount of regularization. Corresponds to the inverse of lambda. E.g. if lambda = 10, then C = 0.1
    :param random_state: Seeding for reproducibility.
    :param tol: Tolerance for stopping criteria.
    :return: Lasso model
    """
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=100, random_state=random_state, tol=tol)
    return model


def elasticnet(C=1.0, l1_ratio=0.0, random_state=0, tol=0.0001):
    """
    L1 and L2 regularized logistic regression model
    :param C: Amount of regularization. Corresponds to the inverse of lambda. E.g. if lambda = 10, then C = 0.1
    :param l1_ratio: Amount of l1 regularization. If l1_ratio = 0.1, then l2_ratio must be 0.9
    :param random_state: Seeding for reproducibility
    :param tol: Tolerance for stopping criteria.
    :return: elasticnet model
    """

    model = LogisticRegression(penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state, tol=tol)
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
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    C = hyperparams['C']

    if model_type == 'lasso':
        results = {'C': C, 'mean_train_score': [], 'mean_val_score': [], 'std_train_score': [], 'std_val_score': [],
                   'best_hyperparam': {}}

        if X_choice == 'all_proteins' and standardize == True:
            for i, c in enumerate(C):  # loop over hyperparameter
                train_aucs, val_aucs = [], []  # for storing aucs over all splits

                for j, (train_index, val_index) in enumerate(rskf.split(X, y)):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                    features_to_scale = list(X_train.columns)
                    features_to_scale = [x for x in features_to_scale if x not in NON_PROT_FEAT]
                     # drop since want to standardize proteins only

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



    elif model_type == 'elasticnet':
        l1_ratio = hyperparams['l1_ratio']

        results = {'C': C, 'l1_ratio': l1_ratio, 'mean_train_score': [], 'mean_val_score': [],
                   'std_train_score': [], 'std_val_score': [], 'best_hyperparam': {}}

        if X_choice == 'all_proteins' and standardize == True:
            for h, l1 in enumerate(l1_ratio):
                for i, c in enumerate(C):  # loop over hyperparameter
                    train_aucs, val_aucs = [], []  # for storing aucs over all splits

                    for j, (train_index, val_index) in enumerate(rskf.split(X, y)):
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                        features_to_scale = list(X_train.columns)
                        features_to_scale = [x for x in features_to_scale if x not in NON_PROT_FEAT]
                        # drop since want to standardize proteins only

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

                        clf = elasticnet(C=c, l1_ratio=l1, tol=0.01)

                        clf.fit(X_train_transf, y_train)

                        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train_transf)[:, 1])
                        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val_transf)[:, 1])

                        train_aucs.append(train_auc)
                        val_aucs.append(val_auc)
                        print(f'Finished {X_choice} -- Hyperparameter {h + 1}/{len(l1_ratio)}, {i + 1}/{len(C)} (l1_ratio={l1}, C={c}): Experiment {j + 1}/{n_splits * n_repeats}')

                    results['mean_train_score'].append(np.mean(train_aucs))  # first 17 values for l1_ratio=0.0, next 17 for l1_ratio=0.1...
                    results['mean_val_score'].append(np.mean(val_aucs))
                    results['std_train_score'].append(np.std(train_aucs))
                    results['std_val_score'].append(np.std(val_aucs))

        else:  # no standardization of protein levels during cross-validation ('baseline')
            for h, l1 in enumerate(l1_ratio):
                for i, c in enumerate(C):  # loop over hyperparameter
                    train_aucs, val_aucs = [], []  # for storing aucs over all splits

                    for j, (train_index, val_index) in enumerate(rskf.split(X, y)):
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                        clf = elasticnet(C=c, l1_ratio=l1, tol=0.01)

                        clf.fit(X_train, y_train)

                        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
                        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

                        train_aucs.append(train_auc)
                        val_aucs.append(val_auc)
                        print(f'Finished {X_choice} -- Hyperparameter {h + 1}/{len(l1_ratio)}, {i + 1}/{len(C)} (l1_ratio={l1}, C={c}): Experiment {j + 1}/{n_splits * n_repeats}')

                    results['mean_train_score'].append(np.mean(train_aucs))
                    results['mean_val_score'].append(np.mean(val_aucs))
                    results['std_train_score'].append(np.std(train_aucs))
                    results['std_val_score'].append(np.std(val_aucs))

        idx_of_best_val_score = results['mean_val_score'].index(max(results['mean_val_score']))

        idx_of_l1_ratio = idx_of_best_val_score // len(C)
        idx_of_C = idx_of_best_val_score % len(C)

        results['best_hyperparam']['l1_ratio'] = results['l1_ratio'][idx_of_l1_ratio]
        results['best_hyperparam']['C'] = results['C'][idx_of_C]

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



def plot_auc(model_type, data, outcome, hyperparams, cv_model_results, colors):
    """
    Plots the training AUC scores as a function of the hyperparameters
    :param model_type: which model results to plot
    :param data: which dataset was used e.g. infe
    :param outcome: The target label
    :param hyperparams: dictionary of hyperparameters
    :param cv_model_results: results from cross-validation
    :param colors: colors to use in plotting
    :return:
    """

    save_dir = os.path.join(PLOTS_DIR, 'sensitivity_analysis1', 'train_auc')
    os.makedirs(save_dir, exist_ok=True)

    labels = ['Baseline Model', 'Protein Model']

    C_range = hyperparams['C']
    # convert to 2 decimal places string then to float
    log_10_lamb = [float("{:.2f}".format(math.log10(1 / c))) if c != 0 else c for c in C_range]

    if model_type == 'lasso':
        file_to_save = f'{save_dir}/{data}_{outcome}_{model_type}_train_auc.png'

        sns.set_theme()
        # Here are some plot styles, which primarily make this plot larger for display purposes.
        plotting_params = {'axes.labelsize': 24,
                           'legend.fontsize': 24,
                           'xtick.labelsize': 24,
                           'ytick.labelsize': 24,
                           'axes.titlesize': 20}
        plt.rcParams.update(plotting_params)
        plt.subplots(figsize=(15, 12))

        for i, choice in enumerate(cv_model_results):  # iterate through keys which are the data set choices

            cv_results = cv_model_results[choice]
            x = log_10_lamb
            y = list(cv_results['mean_val_score'])

            plt.plot(x, y, lw=4, color=colors[i], label=labels[i])

            # get (x,y) at max y value
            ymax = max(y)
            xpos = y.index(ymax)
            xmax = x[xpos]  # convert to actual lambda value
            xmax = 10**xmax
            plt.ylim(top=1)  # set y axis max value to 1

            # Plot a dotted vertical line at the best score for that scorer marked by x
            plt.plot([log_10_lamb[y.index(max(y))]] * 2, np.linspace(0, max(y), 2),
                     linestyle='-.', color=colors[i], marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            if choice == 'fdr_sig_proteins':
                plt.annotate(f'($\lambda$={xmax:.2f}, AUC={ymax:.3f})',
                             (log_10_lamb[y.index(max(y))], max(y) + 0.02))
            else:
                plt.annotate(f'($\lambda$={xmax:.2f}, AUC={ymax:.3f})',
                         (log_10_lamb[y.index(max(y))], max(y) + 0.01), fontsize=20)

            plt.fill_between(log_10_lamb, np.array(cv_results['mean_val_score']) - np.array(cv_results['std_val_score']),
                             np.array(cv_results['mean_val_score']) + np.array(cv_results['std_val_score']),
                             alpha=0.1, color=colors[i])

        plt.ylim(bottom=0.4)
        plt.xlabel('Strength of regularization ($log_{10}(\lambda$))')
        plt.ylabel('Training AUC score')
        plt.legend(bbox_to_anchor=(1, 1))
        # plt.title(f'{data} {outcome} {model_type} training AUC', fontsize=20)
        plt.savefig(file_to_save, bbox_inches='tight')
        plt.show()

    elif model_type == 'elasticnet':

        l1_ratio = hyperparams['l1_ratio']
        l1_ratio = ["{:.2f}".format(ratio) for ratio in l1_ratio]  # prevent long floating points

        for i, choice in enumerate(cv_model_results):  # iterate through keys which are the data set choices

            cv_results = cv_model_results[choice]

            lst = []

            for j in range(len(l1_ratio)):
                l = cv_results['mean_val_score'][17*j:17*(j+1)]
                l.reverse()
                print(l)
                lst.append(l)
            print(lst)
            # df = pd.DataFrame(lst, columns=log_10_lamb[::-1])
            # print(df)
            #
            # sns.set_theme()
            # # Here are some plot styles, which primarily make this plot larger for display purposes.
            # plotting_params = {'axes.labelsize': 18,
            #                    'legend.fontsize': 16,
            #                    'xtick.labelsize': 16,
            #                    'ytick.labelsize': 16,
            #                    'axes.titlesize': 20}
            # plt.rcParams.update(plotting_params)
            # plt.subplots(figsize=(15, 12))
            #
            #
            # sns.heatmap(df, cmap='YlOrBr', annot=True)
            # plt.show()

            for idx, k in enumerate(lst):
                plt.plot(log_10_lamb[::-1], k, lw=1, label=l1_ratio[idx])
            plt.legend(bbox_to_anchor=(1, 1), title="L1 ratio",fancybox=True, shadow=True)
            plt.xlabel('Strength of regularization ($log_{10}(\lambda$))')
            plt.ylabel('Training AUC score')
            plt.xticks(rotation=70)

            # plt.title(f'{data} {outcome} {model_type} training AUC - {labels[i]}')
            file_to_save = f'{save_dir}/{data}_{outcome}_{model_type}_{choice}_train_auc.png'
            plt.savefig(file_to_save, bbox_inches='tight')

            plt.show()


        #     x = log_10_lamb
        #     y = list(cv_results['mean_val_score'])
        #
        #     plt.plot(x, y, lw=4, color=colors[i], label=labels[i])
        #
        #     # get (x,y) at max y value
        #     ymax = max(y)
        #     xpos = y.index(ymax)
        #     xmax = 10 ** (x[xpos])  # convert to actual lambda value
        #     plt.ylim(top=1)  # set y axis max value to 1
        #
        #     # Plot a dotted vertical line at the best score for that scorer marked by x
        #     plt.plot([log_10_lamb[y.index(max(y))]] * 2, np.linspace(0, max(y), 2),
        #              linestyle='-.', color=colors[i], marker='x', markeredgewidth=3, ms=8)
        #
        #     # Annotate the best score for that scorer
        #     if choice == 'fdr_sig_proteins':
        #         plt.annotate(f'($\lambda$={xmax:.2f}, AUC={ymax:.3f})',
        #                      (log_10_lamb[y.index(max(y))], max(y) + 0.02))
        #     else:
        #         plt.annotate(f'($\lambda$={xmax:.2f}, AUC={ymax:.3f})',
        #                      (log_10_lamb[y.index(max(y))], max(y) + 0.01))
        #
        #     plt.fill_between(log_10_lamb,
        #                      np.array(cv_results['mean_val_score']) - np.array(cv_results['std_val_score']),
        #                      np.array(cv_results['mean_val_score']) + np.array(cv_results['std_val_score']),
        #                      alpha=0.1, color=colors[i])
        #
        # plt.ylim(bottom=0.4)
        # plt.xlabel('Strength of regularization ($log_{10}(\lambda$))')
        # plt.ylabel('Training AUC score')
        # plt.legend(bbox_to_anchor=(1, 1))
        # plt.title(f'{data} {outcome} {model_type} training AUC', fontsize=20)
        # plt.savefig(file_to_save, bbox_inches='tight')
        # plt.show()

    else:
        raise NotImplementedError


# plot correlation plots
def plot_correlation(df, y, data, outcome, model_type, prot_list):
    corr_dir = os.path.join(PLOTS_DIR, 'sensitivity_analysis1', 'correlations')
    os.makedirs(corr_dir, exist_ok=True)

    if len(prot_list) > 1000:
        font = 30
        figure_size = (50, 42)
    else:
        font = 10
        figure_size = (10, 8)

    parameters = {'axes.labelsize': font * 2,
                  'legend.fontsize': font * 2,
                  'xtick.labelsize': font,
                  'ytick.labelsize': font,
                  'axes.titlesize': font * 2}
    plt.rcParams.update(parameters)

    cases = df.loc[y == 1]
    controls = df.loc[y == 0]

    plt.subplots(figsize=figure_size)

    # All samples
    df = df[prot_list]
    corr = df.corr(method='spearman').abs()
    sns.heatmap(corr, cmap='Blues')

    # plt.title(f"{data} {outcome} - All Samples (All proteins)")
    plt.ylabel(r'$\leftarrow$ Increasing p value')
    plt.xlabel(r'Increasing p value $\rightarrow $')
    plt.xticks(range(0, len(prot_list)), prot_list, fontsize=6)
    plt.yticks(range(0, len(prot_list)), prot_list, fontsize=6)
    plt.savefig(f'{corr_dir}/{model_type}_{data}_{outcome}_all_samples_spearman_correlation.png', bbox_inches='tight')
    plt.show()

    # Cases
    plt.subplots(figsize=figure_size)

    df = cases[prot_list]
    corr = df.corr(method='spearman').abs()
    sns.heatmap(corr, cmap='Blues')

    #plt.title(f"{data} {outcome} - Cases (All proteins)")
    plt.ylabel(r'$\leftarrow$ Increasing p value')
    plt.xlabel(r'Increasing p value $\rightarrow $')
    plt.xticks(range(0, len(prot_list)), prot_list, fontsize=6)
    plt.yticks(range(0, len(prot_list)), prot_list, fontsize=6)
    plt.savefig(f'{corr_dir}/{model_type}_{data}_{outcome}_cases_spearman_correlation.png', bbox_inches='tight')
    plt.show()

    # Controls
    plt.subplots(figsize=figure_size)

    df = controls[prot_list]
    corr = df.corr(method='spearman').abs()
    sns.heatmap(corr, cmap='Blues')

    #plt.title(f"{data} {outcome} - Controls (All proteins)")
    plt.ylabel(r'$\leftarrow$ Increasing p value')
    plt.xlabel(r'Increasing p value $\rightarrow $')
    plt.xticks(range(0, len(prot_list)), prot_list, fontsize=6)
    plt.yticks(range(0, len(prot_list)), prot_list, fontsize=6)
    plt.savefig(f'{corr_dir}/{model_type}_{data}_{outcome}_controls_spearman_correlation.png', bbox_inches='tight')
    plt.show()


def plot_nonzero_coefficients(type, x_val, y_val, data, outcome, model_type, X_choice, color):
    """
    Plots the coefficient values of each variable in the model if the coefficient is nonzero.
    :param type: Whether to plot nonzero coefficients, sorted nonzero coefficients, or sorted absolute value of the coefficients
    :param x_val: values for the x-axis i.e. the coefficient values
    :param y_val: values for the y-axis i.e. feature variable names corresponding to coefficient values on x-axis
    :param data: the dataset used e.g. infe, non_infe
    :param outcome: the outcome name e.g. A2, A3, B2, C1
    :param model_type:
    :param X_choice: The X dataframe used
    :param color: colors for each plot
    :return:
    """

    coef_dir = os.path.join(PLOTS_DIR, 'sensitivity_analysis1', 'model_coef')
    os.makedirs(coef_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 8))
    if X_choice in ['all_proteins', 'fdr_sig_proteins']:
        ax.tick_params(axis='y', labelsize=6)

    # Plot nonzero coefficients
    y_pos = np.arange(len(y_val))
    ax.barh(y_pos, x_val, align='center', color=color, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_val)
    ax.invert_yaxis()  # labels read top-to-bottom

    if type == 'nonzero_coef':
        ax.set_xlabel('Coefficient values')
        ax.set_ylabel(r'Nonzero model variables')
        # ax.set_title(f'{data} {outcome} {model_type} - {X_choice}: number of nonzero coefficients = {len(x_val)}')

        plt.savefig(f'{coef_dir}/{data}_{outcome}_{model_type}_{X_choice}_nonzero_coef={len(x_val)}.png',
                    bbox_inches='tight')

    elif type == 'sorted_nonzero_coef':
        ax.set_xlabel('Coefficient values')
        ax.set_ylabel(r'Nonzero model variables')
        # ax.set_title(f'Sorted - {data} {outcome} {model_type} - {X_choice}: number of nonzero coefficients = {len(x_val)}')

        plt.savefig(f'{coef_dir}/{data}_{outcome}_{model_type}_{X_choice}_nonzero_coef_sorted={len(x_val)}.png',
                     bbox_inches='tight')

    elif type == 'abs_sorted_nonzero_coef':
        ax.set_xlabel('abs(Coefficient values)')
        ax.set_ylabel(r'Nonzero model variables')
        # ax.set_title(f'Absolute value, Sorted - {data} {outcome} {model_type} - {X_choice}: number of nonzero coefficients = {len(x_val)}')

        plt.savefig(f'{coef_dir}/{data}_{outcome}_{model_type}_{X_choice}_nonzero_coef_abs_sorted={len(x_val)}.png',
             bbox_inches='tight')
    else:
        raise NotImplementedError
    plt.show()


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
    print(df.head())
    print(y)
    # Look at data
    # plot_pca(df=df, y=y, data=data, outcome=outcome, cluster_by='samples', num_components=20)
    # plot_age_distribution(df=df, y=y, data=data, outcome=outcome)
    # plot_protein_level_distribution(df=df, y=y, data=data, outcome=outcome, prot_list=df.columns.tolist()[2:7])  # plot distribution of first 5 proteins

    X_choices = ['baseline', 'all_proteins']

    hyperparams = get_hyperparams(model_type=model_type)
    # print(hyperparams)

    # colors = ['#d53e4f', 'lightcoral', 'blue', 'purple', 'lime', '#fee08b', '#fc8d59']
    colors = ['#d53e4f', 'blue']
    cv_results = {}

    model_dir = os.path.join(ROOT_DIR, 'results', 'models', 'sensitivity_analysis1')
    os.makedirs(model_dir, exist_ok=True)
    cv_results_path = f'{model_dir}/{model_type}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_cv_results.pkl'

    if config['params_search']:  # run hyperparam search and save model results

        os.makedirs(model_dir, exist_ok=True)

        for X_choice in X_choices:
            X = get_samples(df=df, data=data, outcome=outcome, choice=X_choice, fdr=0.01)
            print(X.shape)
            print(X.head())
            cv_result = repeated_stratified_kfold_gridsearchcv(X=X,
                                                                y=y,
                                                                X_choice=X_choice,
                                                                standardize=standardize,
                                                                hyperparams=hyperparams,
                                                                model_type=model_type,
                                                                n_splits=5,
                                                                n_repeats=10,  # 10
                                                                random_state=SEED)
            cv_results[X_choice] = cv_result

        # save model parameters
        with open(cv_results_path, 'wb') as fp:
            pickle.dump(cv_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'...Model results saved at {cv_results_path}')

    # load model parameters
    with open(cv_results_path, 'rb') as fp:
        cv_results = pickle.load(fp)
    print(cv_results.keys())
    print(cv_results)

    plot_auc(model_type=model_type,
                    data=data,
                    outcome=outcome,
                    hyperparams=hyperparams,
                    cv_model_results=cv_results,
                    colors=colors)
    # use best hyperparameter to train on entire dataset

    # create directory for saving final model
    final_model_dir = os.path.join(ROOT_DIR, 'results', 'models', 'sensitivity_analysis1', 'final')
    os.makedirs(final_model_dir, exist_ok=True)

    complete_summary = {}  # for storing the mean and std of each protein
    for i, X_choice in enumerate(X_choices):

        C = cv_results[X_choice]['best_hyperparam']['C']
        clf = lasso(C=C, random_state=SEED)
        X = get_samples(df=df, data=data, outcome=outcome, choice=X_choice, fdr=0.01)

        # save training features
        train_features = list(X.columns)
        train_features_file = f'{final_model_dir}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_train_features.pkl'
        with open(train_features_file, "wb") as fp:  # Pickling
            pickle.dump(train_features, fp)

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

            # for feature in features_to_scale:
            #     summary = {}
            #     summary['mean'] = X[feature].mean(axis=0)
            #     summary['std'] = X[feature].std(axis=0, ddof=0)  # ddof=0 (default = 1) to be consistent with StandardScaler()
            #     complete_summary[feature] = summary  # store dictionary containing mean and std of protein inside dictionary
            #
            # print(complete_summary)

            features_to_scale = [x for x in features_to_scale if x not in NON_PROT_FEAT]

            print(len(features_to_scale))  # features_to_scale should be a list of all 5284 protein names
            # See what model coefficient values are if final training done on standardized vs. nonstandardized data
            X_transf = X.copy()
            features = X_transf[features_to_scale]
            scaler = StandardScaler().fit(features.values)  # fit scaler on X_train; uses same mean and std as complete_summary
            features = scaler.transform(features.values)
            X_transf[features_to_scale] = features

            # print(X_transf.head())  # check that proteins in X (for 'all_proteins') is standardized properly
            clf.fit(X_transf, y)

            # save the scaler dictionary
            # scaler_file = f'{final_model_dir}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_scaler.pkl'
            # pickle.dump(complete_summary, open(scaler_file, 'wb'))

        elif X_choice == 'baseline':  # don't standardize (since don't have proteins) and directly fit on X
            # print(X.head())
            print(X.head())
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

        # save model coefficients
        model_coef_results = dict(zip(nonzero_coef_names, nonzero_coef))
        print(model_coef_results)
        model_coef_file = f'{final_model_dir}/{X_choice}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_coef.pkl'
        with open(model_coef_file, 'wb') as fp:
            pickle.dump(model_coef_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # plot spearman correlation of coefficients
        nonzero_prot_list = [protein for protein in nonzero_coef_names if protein not in ['age_at_diagnosis', 'sex_M', 'ProcessTime', 'SampleGroup']]  # keep only proteins

        if X_choice == 'all_proteins':
            plot_correlation(df=X, y=y, data=data, outcome=outcome, model_type=model_type, prot_list=nonzero_prot_list)
            # plot_correlation(df=X, y=y, data=data, outcome=outcome, prot_list=X.columns.to_list()[4:54])
