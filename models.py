"""
TODO: Description of code
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

import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic
DAT_DIR = os.path.join(ROOT_DIR, 'results', 'datasets')  # .../somalogic/results/datasets
FDR_DAT_DIR = os.path.join(ROOT_DIR, 'results', 'all_proteins', 'age+sex+protlevel')
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


def preprocess(df):
    """
    Takes data and one-hot encodes dummy variables (e.g. sex becomes sex_F, sex_M) and drops
    one of the dummy variables to prevent multicollinearity.
    Next, replaces NA values with the mean of the column.
    Finally, it sorts the columns into order: age_at_diagnosis, sex_M, proteins
    :param df: The dataframe to be processed
    :return: The preprocessed dataframe
    """
    var = df.iloc[:, :2]  # age_at_diagnosis, sex
    prot = df.iloc[:, 2:]  # proteins
    var = pd.get_dummies(var)  # convert categorical variables to dummy variables
    df = pd.concat([var, prot], axis=1)  # merge dataframes by column
    df.drop('sex_F', axis=1, inplace=True)  # drop one of dummy variables to prevent multicollinearity
    df.fillna(df.mean(), inplace=True)  # fill na values with the mean

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
        df = df[['age_at_diagnosis', 'sex_M']]

    elif choice == 'all_proteins':
        df = df

    elif choice == 'fdr_sig_proteins':
        # get list of proteins that survived fdr correction by reading in logistic regression results
        # from prior analysis
        file_path = FDR_DAT_DIR + '/' + '_'.join(['', data, outcome, 'LR',
                                                  'age+sex+protlevel', 'Analysis=all_proteins.xlsx'])
        lr_results = pd.read_excel(file_path, engine='openpyxl')  # without engine, will not open
        lr_results_sig = lr_results[lr_results['FDRp'] < fdr]  # get subset surviving FDR
        fdr_sig_prot_names = list(lr_results_sig["Protein"])

        df = pd.concat([df['age_at_diagnosis'], df['sex_M'], df[fdr_sig_prot_names]], axis=1)

    elif choice == 'ABO':
        df = df[['age_at_diagnosis', 'sex_M', 'ABO.9253.52']]

    elif choice == 'CRP':
        df = df[['age_at_diagnosis', 'sex_M', 'CRP.4337.49']]

    elif choice == 'ABO_CRP':
        df = df[['age_at_diagnosis', 'sex_M', 'ABO.9253.52', 'CRP.4337.49']]

    else:
        raise NotImplementedError

    return df


# def logistic_regression(random_state=0):
#     model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=100, random_state=random_state)
#     return model

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
                                           model_type='lasso',
                                           hyperparams={},
                                           n_splits=5,
                                           n_repeats=10,
                                           random_state=0):
    #TODO: docstring

    # get best lambda, plot AUC curves,

    # Perform Grid Search on each parameter configuration and cross validation
    if model_type == 'lasso':
        model = lasso(random_state)
    elif model_type == 'elasticnet':
        model = elasticnet(random_state)
    else:
        raise NotImplementedError

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    clf = GridSearchCV(estimator=model, param_grid=hyperparams, cv=rskf, scoring='roc_auc', verbose=3,
                       return_train_score=True)  # uses stratified Kfold CV
    clf.fit(X, y)
    print(sorted(clf.cv_results_.keys()))
    cv_results = clf.cv_results_

    print()
    best_params = clf.best_params_
    print(f'Best Parameter: {best_params}')

    print()
    print(f'Mean cross-validated score of the best_estimator: {clf.best_score_}')

    return best_params, cv_results


def get_model_coefficients(clf, X):
    #TODO: docstring

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
    C_range = [10 ** (x / 4) for x in range(-8, 9)]
    print(f'C range: {C_range}')
    l = [math.log10(1 / c) for c in C_range]
    print(f'Lambda range: {l}')

    if model_type == 'lasso':
        hyperparams = {'C': C_range}

    elif model_type == 'elasticnet':
        l1_ratio = [x * 0.1 for x in range(0, 11)]
        hyperparams = {'C': C_range, 'l1_ratio': l1_ratio}

    return hyperparams

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    file_path = DAT_DIR + '/' + '_'.join([data, outcome, 'LR', 'age+sex+protlevel.csv'])

    df = pd.read_csv(file_path, low_memory=False)

    X, y = split_x_y(df, outcome)
    df = preprocess(X)  # age_at_diagnosis, sex_M, proteins
    # Look at data
    #plot_pca(df=df, y=y, data=data, outcome=outcome, cluster_by='samples', num_components=20)
    #plot_age_distribution(df=df, y=y, data=data, outcome=outcome)
    #plot_protein_level_distribution(df=df, y=y, data=data, outcome=outcome, prot_list=df.columns.tolist()[2:7])  # plot distribution of first 5 proteins


    X_choices = ['baseline', 'all_proteins', 'fdr_sig_proteins', 'ABO', 'CRP', 'ABO_CRP']

    model_type = config['model_type']

    hyperparams = get_hyperparams(model_type=model_type)

    colors = ['#d53e4f', 'blue', 'purple', 'lime', '#fee08b', '#fc8d59']
    model_results = {}

    model_dir = os.path.join(ROOT_DIR, 'results', 'models')
    model_results_path = f'{model_dir}/{model_type}_{data}_{outcome}_results.pkl'

    if config['params_search']:  # run hyperparam search and save model results

        os.makedirs(model_dir, exist_ok=True)

        for X_choice in X_choices:

            X = get_samples(df=df, data=data, outcome=outcome, choice=X_choice, fdr=0.01)

            best_params, cv_results = repeated_stratified_kfold_gridsearchcv(X,
                                                                             y,
                                                                             X_choice=X_choice,
                                                                             hyperparams=hyperparams,
                                                                             model_type=model_type,
                                                                             n_splits=5,
                                                                             n_repeats=10,
                                                                             random_state=SEED)
            cv_results['best_params'] = best_params  # nested dictionary e.g. {'best_params': {'C': 1.0}}
            model_results[X_choice] = cv_results

        # save model parameters
        with open(model_results_path, 'wb') as fp:
            pickle.dump(model_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'...Model results saved at {model_results_path}')

    # load model parameters
    with open(model_results_path, 'rb') as fp:
        model_results = pickle.load(fp)
    print(model_results)

    # plot_auc(model_type=model_type,
    #                 data=data,
    #                 outcome=outcome,
    #                 hyperparams=hyperparams,
    #                 model_results=model_results,
    #                 colors=colors)

    # use best hyperparameter to train on entire dataset
    for i, X_choice in enumerate(X_choices):

        C = model_results[X_choice]['best_params']['C']
        clf = lasso(C=C, random_state=SEED)
        X = get_samples(df=df, data=data, outcome=outcome, choice=X_choice, fdr=0.01)

        # plot CV ROC
        #####################################

        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

        fig, ax = plt.subplots(figsize=(10, 8))

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        cf_matrix = np.zeros((2, 2))
        for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            lr = lasso(C=C, random_state=SEED)

            lr.fit(X_train, y_train)

            viz = plot_roc_curve(lr, X_test, y_test,
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=f"{X_choice} - Receiver operating characteristic")
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
        #           fancybox=True, shadow=True, ncol=3, prop={'size':5})
        ax.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, prop={'size': 6})
        plt.show()
        #
        #
        #
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

        assert False

        ######################################
        clf.fit(X, y)

        coef, coef_names, nonzero_coef, nonzero_coef_names, \
        sorted_nonzero_coef, sorted_nonzero_coef_names, abs_sorted_nonzero_coef, abs_sorted_nonzero_coef_names = get_model_coefficients(clf=clf, X=X)

        # plot nonzero coefficient values to determine the effect sizes
        plot_nonzero_coefficients(type='nonzero_coef', x_val=nonzero_coef, y_val=nonzero_coef_names,
                                  data=data, outcome=outcome, model_type=model_type, X_choice=X_choice, color=colors[i])

        plot_nonzero_coefficients(type='sorted_nonzero_coef', x_val=sorted_nonzero_coef, y_val=sorted_nonzero_coef_names,
                                  data=data, outcome=outcome, model_type=model_type, X_choice=X_choice, color=colors[i])

        plot_nonzero_coefficients(type='abs_sorted_nonzero_coef', x_val=abs_sorted_nonzero_coef, y_val=abs_sorted_nonzero_coef_names,
                                  data=data, outcome=outcome, model_type=model_type, X_choice=X_choice, color=colors[i])

        prot_list = [protein for protein in nonzero_coef_names if protein not in ['age_at_diagnosis', 'sex_M']]  # keep only proteins

        if X_choice == 'all_proteins':
            plot_correlation(df=X, y=y, data=data, outcome=outcome, prot_list=prot_list)
