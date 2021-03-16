import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os
import numpy as np
import math

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic

# Plot Number of coefficients left as a function of strength of regularization

# confusion matrix
# individual AUC curves for all 50 experiments with best AUC curve darker than rest


def plot_pca(df, y, data,  outcome, cluster_by='samples', num_components=20):
    #TODO: docstring

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

        colors = ['cyan', 'magenta']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1], target_names):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title(f'PCA of {data} {outcome} clustered by {cluster_by}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{cluster_by}_pca.png',
                    bbox_inches='tight')
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
    plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{cluster_by}_pca_variance.png',
                bbox_inches='tight')
    plt.show()

    ## Save eigenvector of PCs
    # col_names = ['PC'+ str(i) for i in range(1, num_components+1) ]
    # df = pd.DataFrame(data=X_r, index=None, columns=col_names)

    # df.to_csv(f"./{data}_{outcome}_{cluster_by}_pcs.csv")


def plot_auc(model_type, data, outcome, hyperparams, model_results, colors):
    #TODO: Add docstring

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
        C_range = hyperparams['C']
        lamb = [math.log10(1 / c) if c != 0 else c for c in C_range]
        l1_ratio = hyperparams['l1_ratio']


    else:
        raise NotImplementedError



def plot_nonzero_coefficients(type, x_val, y_val, data, outcome, model_type, X_choice, color):
    #TODO: add docstring

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