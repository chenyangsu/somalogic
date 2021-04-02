import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic

model_dir = os.path.join(ROOT_DIR, 'results', 'models')

sns.set_theme()
# Here are some plot styles, which primarily make this plot larger for display purposes.
plotting_params = {'axes.labelsize': 18,
              'legend.fontsize': 16,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'axes.titlesize': 20}
plt.rcParams.update(plotting_params)
plt.subplots(figsize=(15, 12))

colors = ['blue', 'red', 'green', 'blue', 'red', 'green']
for i in range(6):
    model_type, soma_data, nat_log_transf, standardize, data, outcome = 'lasso', 'normalized', False, False, 'infe', 'A2'

    if i == 0:
        model_type, soma_data, nat_log_transf, standardize, data, outcome = 'lasso', 'normalized', True, True, 'infe', 'A2'
    elif i == 1:
        model_type, soma_data, nat_log_transf, standardize, data, outcome = 'lasso', 'normalized', True, False, 'infe', 'A2'
    elif i == 2:
        model_type, soma_data, nat_log_transf, standardize, data, outcome = 'lasso', 'normalized', False, False, 'infe', 'A2'
    # unnormalized
    elif i == 3:
        model_type, soma_data, nat_log_transf, standardize, data, outcome = 'lasso', 'unnormalized', True, True, 'infe', 'A2'
    elif i == 4:
        model_type, soma_data, nat_log_transf, standardize, data, outcome = 'lasso', 'unnormalized', True, False, 'infe', 'A2'
    elif i == 5:
        model_type, soma_data, nat_log_transf, standardize, data, outcome = 'lasso', 'unnormalized', False, False, 'infe', 'A2'


    model = f'{model_dir}/{model_type}-soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}_{data}_{outcome}_results.pkl'
    # model = f'{model_dir}/lasso-soma_data=normalized-nat_log_transf=False-standardize=False_infe_A2_results111.pkl'

    # load model parameters
    with open(model, 'rb') as fp:
        model_results = pickle.load(fp)

    C_range = model_results['C']
    print(f'C range: {C_range}')
    log_10_lamb = [math.log10(1 / c) if c != 0 else c for c in C_range]
    print(f'Lambda range: {log_10_lamb}')

    choice = f'soma_data={soma_data}-nat_log_transf={nat_log_transf}-standardize={standardize}'
    cv_results = model_results
    print(cv_results.keys())

    x = log_10_lamb
    y = list(cv_results['mean_val_score'])

    if i < 3:
        plt.plot(x, y, lw=4, color=colors[i], label=f'{choice}')
    elif i < 6:
        plt.plot(x, y, lw=4, linestyle='dotted', color=colors[i], label=f'{choice}')

    # get (x,y) at max y value
    ymax = max(y)
    xpos = y.index(ymax)
    xmax = x[xpos]
    xmax = 10**xmax
    plt.ylim(top=1)  # set y axis max value to 1

    # Plot a dotted vertical line at the best score for that scorer marked by x
    plt.plot([log_10_lamb[y.index(max(y))]] * 2, np.linspace(0, max(y), 2),
             linestyle='-.', color=colors[i], marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    if i == 0:
        plt.annotate(f'($\lambda={xmax:.2f}$, AUC={ymax:.3f})',
                     (log_10_lamb[y.index(max(y))], max(y) + 0.03))
    elif i == 1:
        plt.annotate(f'($\lambda={xmax:.2f}$, AUC={ymax:.3f})',
                     (log_10_lamb[y.index(max(y))]-0.5, max(y) + 0.01))
    elif i == 2:
        plt.annotate(f'($\lambda={xmax:.2f}$, AUC={ymax:.3f})',
                     (log_10_lamb[y.index(max(y))]-0.1, max(y) + 0.02))
    elif i == 4:
        plt.annotate(f'($\lambda={xmax:.2f}$, AUC={ymax:.3f})',
                     (log_10_lamb[y.index(max(y))] - 0.5, max(y) + 0.02))
    else:
        plt.annotate(f'($\lambda={xmax:.2f}$, AUC={ymax:.3f})',
                 (log_10_lamb[y.index(max(y))], max(y) + 0.01))

    plt.fill_between(log_10_lamb, np.array(cv_results['mean_val_score']) - np.array(cv_results['std_val_score']),
                     np.array(cv_results['mean_val_score']) + np.array(cv_results['std_val_score']),
                     alpha=0.1, color=colors[i])

plt.ylim(bottom=0.4)
plt.xlabel('Strength of regularization ($log_{10}(\lambda$))')
plt.ylabel('AUC score')
plt.legend(loc='best')
plt.title(f'{data} {outcome} {model_type} age+sex+5284proteins', fontsize=20)
# plt.savefig(f'{ROOT_DIR}/results/plots/{data}_{outcome}_{model_type}_auc.png',
#             bbox_inches='tight')
plt.show()
print("DONE")