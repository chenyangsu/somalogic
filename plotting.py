"""
TODO: Description of code
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os
import numpy as np
import math
import pandas as pd
from sklearn.metrics import auc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic

# Plot Number of coefficients left as a function of strength of regularization

# confusion matrix
# individual AUC curves for all 50 experiments with best AUC curve darker than rest


def plot_age_distribution(df, y, data, outcome):
    """
    Plots age distribution violin plot of cases vs. controls and males vs. females
    :param df: The dataframe containing the samples and feature for age
    :param data: the dataset used
    :param y: the column of outcomes
    :param outcome: the outcome used e.g. A2, A3, B2, C1
    :return: a scatter plot
    """
    sns.set_theme(style="whitegrid")
    dataframe = pd.concat([df, y], axis=1)
    dataframe[outcome] = dataframe[outcome].map({1: 'Case', 0: 'Control'})  # replace 1 to case, 0 to control
    dataframe['sex_M'] = dataframe['sex_M'].map({1: 'M', 0: 'F'})  # replace 1 to case, 0 to control

    ax = sns.violinplot(x=outcome, y='age_at_diagnosis', hue='sex_M',
                        palette=['tomato', 'dodgerblue'], data=dataframe)

    ax.set_ylabel('Age')
    ax.set_xlabel(f'{outcome} Outcome')
    ax.legend(loc='best')
    ax.set_title(f'{data} {outcome} age distribution')
    plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_age_violin.png', bbox_inches='tight')
    plt.show()

# plots age distribution as a scatter plot by cases/controls
# def plot_age_distribution(df, y, data, outcome):
#     """
#     Plots age distribution scatter plot of cases vs. controls
#     :param df: The dataframe containing the samples and feature for age
#     :param data: the dataset used
#     :param y: the column of outcomes
#     :param outcome: the outcome used e.g. A2, A3, B2, C1
#     :return: a scatter plot
#     """
#     plt.subplots(figsize=(10, 8))
#     cases = df[y == 1]
#     controls = df[y == 0]
#
#     plt.scatter(list(range(cases.shape[0])), cases['age_at_diagnosis'], color='magenta', label='Cases')
#     plt.scatter(list(range(controls.shape[0])), controls['age_at_diagnosis'], color='deepskyblue', label='Controls')
#
#     # plt.scatter
#     plt.ylabel("Age")
#     plt.xlabel("samples")
#     plt.legend(loc='best')
#     plt.title(f'{data} {outcome} age distribution')
#     plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_age_scatter.png',
#                     bbox_inches='tight')
#     plt.show()


# plot histogram
def plot_protein_level_distribution(df, y, data, outcome, prot_list):
    """
    Plots protein level distribution in a violin plot by cases and controls and splitting into Male, Female
    :param df: The design matrix
    :param data: The dataset used e.g. infe, non_infe
    :param y: the column of outcomes
    :param outcome: the outcome of interest e.g. A2, A3, B2, C1
    :param prot_list: the list of proteins who
    :return:
    """
    sns.set_theme(style="whitegrid")

    dataframe = pd.concat([df, y], axis=1)
    dataframe[outcome] = dataframe[outcome].map({1: 'Case', 0: 'Control'})  # replace 1 to case, 0 to control
    dataframe['sex_M'] = dataframe['sex_M'].map({1: 'M', 0: 'F'})  # replace 1 to case, 0 to control

    for i in range(len(prot_list)):
        ax = sns.violinplot(x=outcome, y=prot_list[i], hue='sex_M',
                            palette=['tomato', 'dodgerblue'], data=dataframe)

        ax.set_title(f'{data} {outcome} {prot_list[i]}')
        ax.set_xlabel(f'{outcome} Outcome')
        ax.set_ylabel('Protein level')
        ax.legend(loc='best', shadow=False, scatterpoints=1)
        plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{prot_list[i]}_violin.png', bbox_inches='tight')
        plt.show()


# # plot protein levels as 3 histograms overlaying on top of one another
# def plot_protein_level_distribution(df, y, data, outcome, prot_list):
#     """
#     Plots protein level distribution in a histogram by all samples, cases, and controls
#     :param df: The design matrix
#     :param data: The dataset used e.g. infe, non_infe
#     :param y: the column of outcomes
#     :param outcome: the outcome of interest e.g. A2, A3, B2, C1
#     :param prot_list: the list of proteins who
#     :return:
#     """
#     alpha = 0.5
#
#     cases_df = df[y == 1]
#     controls_df = df[y == 0]
#
#
#     for i in range(len(prot_list)):
#         d = df[prot_list[i]].dropna()  # drop NaN values
#         cases = cases_df[prot_list[i]].dropna()  # drop NaN values
#         controls = controls_df[prot_list[i]].dropna()  # drop NaN values
#
#         plt.hist(d, color='black', bins=40, histtype='step', alpha=alpha, label='All samples')
#         plt.hist(cases, color='deepskyblue', bins=40, histtype='step', alpha=alpha, label='Cases')
#         plt.hist(controls, color='magenta', bins=40, histtype='step', alpha=alpha, label='Controls')
#
#         plt.title(f'{data} {outcome} {prot_list[i]}')
#         plt.xlabel('Protein level')
#         plt.ylabel('Number of protein samples')
#         plt.legend(loc='best', shadow=False, scatterpoints=1)
#         plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{prot_list[i]}_hist.png', bbox_inches='tight')
#         plt.show()


# plot correlation plots
def plot_correlation(df, y, data, outcome, prot_list):
    #TODO: docstring
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

    plt.title(f"{data} {outcome} - All Samples (All proteins)")
    plt.ylabel(r'$\leftarrow$ Increasing p value')
    plt.xlabel(r'Increasing p value $\rightarrow $')
    #plt.savefig(f'{data}_{outcome}_all_samples_spearman_correlation.png')
    plt.show()

    # Cases
    plt.subplots(figsize=figure_size)

    df = cases[prot_list]
    corr = df.corr(method='spearman').abs()
    sns.heatmap(corr, cmap='Blues')

    plt.title(f"{data} {outcome} - Cases (All proteins)")
    plt.ylabel(r'$\leftarrow$ Increasing p value')
    plt.xlabel(r'Increasing p value $\rightarrow $')
    #plt.savefig(f'{data}_{outcome}_cases_spearman_correlation.png')
    plt.show()

    # Controls
    plt.subplots(figsize=figure_size)

    df = controls[prot_list]
    corr = df.corr(method='spearman').abs()
    sns.heatmap(corr, cmap='Blues')

    plt.title(f"{data} {outcome} - Controls (All proteins)")
    plt.ylabel(r'$\leftarrow$ Increasing p value')
    plt.xlabel(r'Increasing p value $\rightarrow $')
    #plt.savefig(f'{data}_{outcome}_controls_spearman_correlation.png')
    plt.show()

# plot spearman correlation between proteins
def plot_pca(df, y, data,  outcome, cluster_by='samples', num_components=20):
    """
    Plots two figures: PC1 vs. PC2 clustering and a scree plot showing the variances explained per principal component
    :param df: design matrix containing features of interest e.g. age, sex, proteins
    :param y: outcomes
    :param data: the dataset used e.g. infe, non_infe
    :param outcome: the outcome name e.g. A2, A3, B2, C1
    :param cluster_by: whether to cluster by samples or features
    :param num_components:  the number of principal components
    :return:
    """

    # fill NaNs with column means or else PCA gets error
    df.fillna(df.mean(), inplace=True)  # fill na values with the mean

    col_names = df.columns.to_list()  # get column names: age_at_diagnosis, sex_M, protein_1, ..., protein_5284
    proteins = col_names[2:]
    X = df[proteins]  # get protein columns
    target_names = ['Controls', 'Cases']

    parameters = {'axes.labelsize': 18,
                  'legend.fontsize': 16,
                  'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'axes.titlesize': 20}
    plt.rcParams.update(parameters)
    sns.set_theme()

    if cluster_by == 'samples':

        pca = PCA(n_components=num_components)
        X_r = pca.fit(X).transform(X)  # eigenvectors

        plt.subplots(figsize=(10, 8))

        colors = ['deepskyblue', 'magenta']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1], target_names):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title(f'PCA of {data} {outcome} clustered by {cluster_by}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{cluster_by}_pca.png', bbox_inches='tight')
        plt.show()

    else:
        raise NotImplementedError

    plt.figure()
    plt.subplots(figsize=(10, 8))
    x = list(range(1, num_components + 1))
    plt.bar(x, pca.explained_variance_ratio_, color='#c51b7d', edgecolor='black')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(ticks=x)  # set xticks to integer values corresponding to PC component
    plt.title(f'Scree Plot of {data} {outcome} clustered by {cluster_by}')
    plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{cluster_by}_pca_variance.png', bbox_inches='tight')
    plt.show()

    ## Save eigenvector of PCs
    # col_names = ['PC'+ str(i) for i in range(1, num_components+1) ]
    # df = pd.DataFrame(data=X_r, index=None, columns=col_names)

    # df.to_csv(f"./{data}_{outcome}_{cluster_by}_pcs.csv")


def plot_auc(model_type, data, outcome, hyperparams, model_results, colors):
    #TODO: Add docstring

    sns.set_theme()
    # Here are some plot styles, which primarily make this plot larger for display purposes.
    plotting_params = {'axes.labelsize': 18,
                  'legend.fontsize': 16,
                  'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'axes.titlesize': 20}
    plt.rcParams.update(plotting_params)
    plt.subplots(figsize=(15, 12))

    if model_type == 'lasso':
        C_range = hyperparams['C']
        lamb = [math.log10(1 / c) if c != 0 else c for c in C_range]

        for i, choice in enumerate(model_results):  # iterate through keys which are the data set choices
            cv_results = model_results[choice]
            x = lamb
            y = list(cv_results['mean_test_score'])

            plt.plot(x, y, lw=4, color=colors[i], label=f'{choice}')

            # get (x,y) at max y value
            ymax = max(y)
            xpos = y.index(ymax)
            xmax = x[xpos]
            plt.ylim(top=1)  # set y axis max value to 1

            # Plot a dotted vertical line at the best score for that scorer marked by x
            plt.plot([lamb[y.index(max(y))]] * 2, np.linspace(0, max(y), 2),
                     linestyle='-.', color=colors[i], marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            plt.annotate(f'($\lambda$={xmax:.2f}, AUC={ymax:.3f})',
                         (lamb[y.index(max(y))], max(y) + 0.01))

            plt.fill_between(lamb, cv_results['mean_test_score'] - cv_results['std_test_score'],
                             cv_results['mean_test_score'] + cv_results['std_test_score'],
                             alpha=0.1, color=colors[i])

        plt.ylim(bottom=0.4)
        plt.xlabel('Strength of regularization ($log_{10}(\lambda$))')
        plt.ylabel('AUC score')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.title(f'{data} {outcome} {model_type}', fontsize=20)
        plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{model_type}_auc.png',
                    bbox_inches='tight')
        plt.show()

    elif model_type == 'elasticnet':
        # C_range = hyperparams['C']
        # lamb = [math.log10(1 / c) if c != 0 else c for c in C_range]
        # l1_ratio = hyperparams['l1_ratio']
        pass

    else:
        raise NotImplementedError


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

    sns.set_theme()

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
        ax.set_title(f'{data} {outcome} {model_type} - {X_choice}: number of nonzero coefficients = {len(x_val)}')

        plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{model_type}_{X_choice}_nonzero_coef.png',
                    bbox_inches='tight')

    elif type == 'sorted_nonzero_coef':
        ax.set_xlabel('Coefficient values')
        ax.set_ylabel(r'Nonzero model variables')
        ax.set_title(
            f'Sorted - {data} {outcome} {model_type} - {X_choice}: number of nonzero coefficients = {len(x_val)}')

        plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{model_type}_{X_choice}_nonzero_coef_sorted.png',
                    bbox_inches='tight')

    elif type == 'abs_sorted_nonzero_coef':
        ax.set_xlabel('abs(Coefficient values)')
        ax.set_ylabel(r'Nonzero model variables')
        ax.set_title(
            f'Absolute value, Sorted - {data} {outcome} {model_type} - {X_choice}: number of nonzero coefficients = {len(x_val)}')

        plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{model_type}_{X_choice}_nonzero_coef_abs_sorted.png',
            bbox_inches='tight')
    else:
        raise NotImplementedError
    plt.show()