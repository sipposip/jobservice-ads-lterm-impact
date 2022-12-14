"""
implementation of the simple model described in the manuscript.

Author: Sebastian Scher (sscher@know-center.at)

"""

import os

from pylab import plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model, metrics

sns.set_context('notebook', font_scale=1)
sns.set_palette('colorblind')
plotdir = './plots/'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)

plt.ioff()

def savefig(fname):
    plt.savefig(plotdir + fname + '.pdf')
    plt.savefig(plotdir + fname + '.svg')


def initialize_data(alpha_prot, maxval, alpha_lb, rand_seed=None):
    """
    generate the initial dataframe
    :param alpha_prot: how much alpha_prot influences x2
    :param maxval: if not None, truncated normal distributions are used, truncated at +-maxval
    :return: dataframe
    """
    np.random.seed(rand_seed)
    # we have two skill features x1 and x2, and a protected feature x_prot
    # the protected feature is correlated with x2
    # we draw them from a truncated normal distribution if maxval is not None
    x_prot = np.random.choice([0, 1], size=n, replace=True)
    if maxval is not None:
        x1 = stats.truncnorm.rvs(-maxval, maxval, size=n)
        x2 = 1 / 2 * (alpha_prot * (x_prot - 0.5) + stats.truncnorm.rvs(-maxval, maxval, size=n))
    else:
        x1 = stats.norm.rvs(size=n)
        x2 = 1 / 2 * (alpha_prot * (x_prot - 0.5) + stats.truncnorm.rvs(size=n))
    # the real skill is simply the mean of x1 and x2
    s_real, s_eff = compute_skill(x1, x2, x_prot, alpha_lb)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x_prot': x_prot, 's_real': s_real, 's_eff': s_eff})
    return df


def compute_skill(x1, x2, x_prot, alpha_lb):
    s_real = (x1 + x2) / 2
    s_eff = s_real + alpha_lb * 2 * (x_prot - 0.5)  # factor to so that we get -1 and 1 from x_prot
    return s_real, s_eff


# logistic regression
def train_model(X, y):
    return linear_model.LogisticRegression().fit(X, y)


def train_and_predict(df, modeltype):
    """
    train the specified modeltype on the dataframe, and make predictions on the training data
    :return: (predictions, coefficients)
    """
    # if there is only once class left in the labels, we make a constant prediction
    # which equals to leaving the labels all the same, and we return 0 for the coefficients
    y = df['class']
    if modeltype == 'full':
        X = df[['x1', 'x_prot']]
    elif modeltype == 'base':
        X = df[['x1']]
    else:
        raise ValueError(f'modeltype {modeltype} not known')
    if len(np.unique(y)) == 1:
        return y, np.array([[0, 0]])
    else:
        model = linear_model.LogisticRegression().fit(X, y)
        return model.predict(X), model.coef_


def real_decision_function(df):
    """compute classes defined by the labor market
    defined via s_eff"""
    s = df['s_eff']
    if decision_function == 'const':
        return (s > 0).astype(int)
    elif decision_function == 'adaptive':
        return (s > s.mean()).astype(int)
    else:
        raise ValueError(f'decision function {decision_function} not known')


def intervention_model(x1, x2, real_class, pred_class, k_matrix):
    """
        update features based on intervention models. works with scalars and vectors
    """
    # in order to have the values in the k_matrix in a nicer format/scale, we
    # multiply it by a constant factor.
    scale_factor = 1 / 50
    k_matrix = k_matrix * scale_factor
    # we have four parameters, for four cases (real class can be 1 or 2, and real class can be 1 or two)
    k_rcl1_pcl1 = k_matrix[0, 0]
    k_rcl1_pcl2 = k_matrix[0, 1]
    k_rcl2_pcl1 = k_matrix[1, 0]
    k_rcl2_pcl2 = k_matrix[1, 1]
    x1_max = 2
    x2_max = 2
    # make an array with k values for the correct real_class - pred_class combination, then we can use
    # vector operations and avoid a slow for loop
    kvec = np.zeros_like(x1)
    kvec[(real_class == 0) & (pred_class == 0)] = k_rcl1_pcl1
    kvec[(real_class == 0) & (pred_class == 1)] = k_rcl1_pcl2
    kvec[(real_class == 1) & (pred_class == 0)] = k_rcl2_pcl1
    kvec[(real_class == 1) & (pred_class == 1)] = k_rcl2_pcl2

    # if x1 or x2 are already above the maximum value that the intervention model can attain,
    # then we do not change them (otherwise they would be reduced when using the same formula)
    # in vector formulation we can achieve this with the np.maximum function (and not the np.max function)
    x1_new = np.maximum(x1 + (x1_max - x1) * kvec, x1)
    x2_new = np.maximum(x2 + (x2_max - x2) * kvec, x2)

    return x1_new, x2_new


def step_model(df, k_matrix, modeltype, alpha_lb):
    """make one step with the complete model (prediction model plus intervention model)
    and update the dataframe"""
    df['x1'], df['x2'] = intervention_model(df['x1'], df['x2'], df['class'], df['class_pred'], k_matrix)
    # compute new real classes
    s_real, s_eff = compute_skill(df['x1'], df['x2'], df['x_prot'], alpha_lb)
    df['s_real'] = s_real
    df['s_eff'] = s_eff
    df['class'] = real_decision_function(df)
    df['class_pred'], coefs = train_and_predict(df, modeltype)

    return df.copy(), coefs


def run_experiment(df_init, n_steps, k_matrix, modeltype, alpha_lb, plot=False):
    res = []
    res_summary = []
    df = df_init.copy()
    # initial predictions
    # compute real and predicted classes
    df['class'] = real_decision_function(df)
    df['class_pred'], _ = train_and_predict(df, modeltype)
    df_init = df  # for plotting later on
    for i in range(n_steps):
        df, coefs = step_model(df, k_matrix, modeltype, alpha_lb)
        # compute summary metrics on data
        summary = {
            'x1_mean': df['x1'].mean(),
            'x2_mean': df['x2'].mean(),
            's_mean': df['s_real'].mean(),
            'x1_group1': df['x1'][df['x_prot'] == 0].mean(),
            'x2_group1': df['x2'][df['x_prot'] == 0].mean(),
            'x1_group2': df['x1'][df['x_prot'] == 1].mean(),
            'x2_group2': df['x2'][df['x_prot'] == 1].mean(),
            's_real_group1': df['s_real'][df['x_prot'] == 0].mean(),
            's_real_group2': df['s_real'][df['x_prot'] == 1].mean(),
            's_eff_group1': df['s_eff'][df['x_prot'] == 0].mean(),
            's_eff_group2': df['s_eff'][df['x_prot'] == 1].mean(),
            'class2_group1': (df['class'][df['x_prot'] == 0]).mean(),
            'class2_group2': (df['class'][df['x_prot'] == 1]).mean(),
        }
        summary['BGSD_real'] = summary['s_real_group2'] - summary['s_real_group1']
        summary['BGSD_eff'] = summary['s_eff_group2'] - summary['s_eff_group1']
        if modeltype == 'full':
            summary['coef2'] = coefs[0, 1],
        summary = pd.DataFrame(summary, index=[i])
        res.append(df)
        res_summary.append(summary)

    res_summary = pd.concat(res_summary)
    df_final = df.copy()

    if plot:
        p = sns.jointplot(x='x1', y='x2', data=df_init, hue='x_prot')
        sns.catplot(x='x_prot', hue='class', data=df_init, kind='count')
        plt.tight_layout()

        # we ue the same  x1 and x2 limits for the final as for the initial data
        sns.jointplot(x='x1', y='x2', data=df_final, hue='x_prot', xlim=p.ax_marg_x.get_xlim(),
                      ylim=p.ax_marg_y.get_ylim())
        sns.catplot(x='x_prot', hue='class', data=df_final, kind='count')
        plt.tight_layout()

        plt.figure()
        ax = plt.subplot(221)
        res_summary[['x1_mean', 'x2_mean']].plot(alpha=0.8, ax=ax)
        ax = plt.subplot(222)
        res_summary[['x1_group1', 'x2_group1', 'x1_group2', 'x2_group2']].plot(alpha=0.8, ax=ax)
        ax = plt.subplot(223)
        res_summary[['s_mean', 's_group1', 's_group2']].plot(alpha=0.8, ax=ax);
        plt.xlabel('t')
        if modeltype == 'full':
            ax = plt.subplot(224)
            res_summary[['coef2']].plot(alpha=0.8, ax=ax)
            plt.suptitle(f'decision_function: {decision_function}, k={k_matrix}');
            plt.xlabel('t')

    return res_summary, res


# generate data
rand_seed = 57654  # fixed random seed for reproducibility
n = 1000
alpha_prot = 2  # influence of alpha_prot on x2
maxval = 2
n_steps = 600  # 600 for adaptive decision function, 100 for constant
alpha_lb = 0  # attention: if this is changed ,the data needs to be initialized again!
decision_function = 'const'  # 'const' | 'adaptive'
df_init = initialize_data(alpha_prot=alpha_prot, maxval=maxval, alpha_lb=alpha_lb, rand_seed=rand_seed)
p = sns.jointplot(x='x1', y='x2', data=df_init, hue='x_prot')
savefig(f'initial_data_alpha_prot{alpha_prot}_maxval{maxval}')
sns.displot(df_init, x='s_real', hue='x_prot')
savefig(f'initial_data_alpha_prot{alpha_prot}_maxbal{maxval}_sreal')
sns.displot(df_init, x='s_eff', hue='x_prot')
plt.title(r'$\alpha_{pr}=$' + str(alpha_prot))
savefig(f'initial_data_alpha_prot{alpha_prot}_maxbal{maxval}_s_eff_alphalb_{alpha_lb}')

# matrix with intervention parameters for the four cases (real class can be 1 or 2, and real class can be 1 or two)
# [[real class 1 & pred class 1, real class 1 & pred class 2],
#  [real class 2 & pred class 1, real class 2 & pred class 2]]

# ----- main experiments
# the configs for the main experiments are collected here and put in a list
configs = [
    {'scenario': "1",
     'description': 'no targeting, no class-dependent effect',
     'k_matrix': np.array([[1, 1],
                           [1, 1]]),
     },
    {'scenario': "2a",
     'description': 'no targeting, class-dependent effect (more on lowprospect group)',
     'k_matrix': np.array([[2, 2],
                           [1 / 2, 1 / 2]]),
     },
    {'scenario': "2b",
     'description': 'no targeting, class-dependent effect (more on highprospect group)',
     'k_matrix': np.array([[1 / 2, 1 / 2],
                           [2, 2]]),
     },
    {'scenario': "3a",
     'description': 'targeting (more on lowprospect group), no class-dependent effect',
     'k_matrix': np.array([[2, 1 / 2],
                           [2, 1 / 2]]),
     },
    {'scenario': "3b",
     'description': 'targeting (more on highprospect group), no class-dependent effect',
     'k_matrix': np.array([[1 / 2, 2],
                           [1 / 2, 2]]),
     },
    {'scenario': "4a",
     'description': 'targeting (more on lowprospect group), class-dependent effect',
     'k_matrix': np.array([[4, 1],
                           [1, 2]]) * 8 ** (-1 / 4),  # this factor is necessary to ensure that the gemoetric mean is 1
     },
    {'scenario': "4b",
     'description': 'targeting (more on lowprospect group), class-dependent effect',
     'k_matrix': np.array([[2, 1],
                           [1, 4]]) * 8 ** (-1 / 4),  # this factor is necessary to ensure that the gemoetric mean is 1
     },
]

# run all experiments
exp_results = {}  # results per experiment
# aggregated results (singl number for all timesteps), we will store it for all experiments in a single
# dataframe,
df_agg = pd.DataFrame()
for config in configs:
    _res = {}
    res_agg = {}
    for modeltype in 'full', 'base':
        res_summary, res = run_experiment(df_init, n_steps=n_steps, k_matrix=config['k_matrix'], modeltype=modeltype,
                                          alpha_lb=alpha_lb)
        _res[modeltype] = {**config, 'res_summary': res_summary,
                           'res': res}

        # compute time after all underpriviliged are classified as high-prospect (and thus the fraction of
        # underpvivilged classified as high prospect (class2_group1) reaches 1
        if np.any(res_summary['class2_group1']) == 1:
            # argmax returns the first True occurence
            res_agg['t_all_upriv_highpros'] = np.argmax(res_summary['class2_group1'] == 1)
        else:
            # if we come here, thatn
            res_agg['t_all_upriv_highpros'] = np.inf

        exp_results[config['scenario']] = _res
        df_agg = df_agg.append(
            pd.DataFrame({'scenario': config['scenario'], **res_agg, 'modeltype': modeltype}, index=[0]))

# -- plot results

# 2-panel plot with between group skill difference and fraction of unpriviliged members classified
# as high-prospect. All experiments in one plot, with each pair of experiments (both modeltypes) a different color.
# the experiments with the full models are plotted with continous, the experiments with the base model with dotted
# lines
figsize=(6,3)
plt.figure(figsize=figsize)
colors = sns.color_palette('colorblind', n_colors=7)
for i, scenario in enumerate(exp_results):
    data_base = exp_results[scenario]['base']['res_summary']
    data_full = exp_results[scenario]['full']['res_summary']
    plt.plot(data_full['BGSD_real'], label=scenario, linestyle='-', color=colors[i])
    plt.plot(data_base['BGSD_real'], linestyle='--', color=colors[i])
plt.legend()
plt.ylabel('$BGSD_{real}$')
sns.despine()
plt.suptitle(r'$\alpha_{pr}=$' + str(alpha_prot) + r' $\alpha_{lb}=$' + str(alpha_lb) +
             'decision_func=' + decision_function)
savefig(f'BGSD_real_{decision_function}_alpha_lb{alpha_lb}')

plt.figure(figsize=figsize)
for i, scenario in enumerate(exp_results):
    data_base = exp_results[scenario]['base']['res_summary']
    data_full = exp_results[scenario]['full']['res_summary']
    plt.plot(data_full['class2_group1'], label=scenario, linestyle='-', color=colors[i])
    plt.plot(data_base['class2_group1'], linestyle='--', color=colors[i])
plt.legend()
plt.ylabel('$FUHP$')
sns.despine()
plt.suptitle(r'$\alpha_{pr}=$' + str(alpha_prot) + r' $\alpha_{lb}=$' + str(alpha_lb) +
             'decision_func=' + decision_function)
savefig(f'FUHP_{decision_function}_alpha_lb{alpha_lb}')

plt.figure(figsize=figsize)
for i, scenario in enumerate(exp_results):
    data_base = exp_results[scenario]['base']['res_summary']
    data_full = exp_results[scenario]['full']['res_summary']
    plt.plot(data_full['BGSD_eff'], label=scenario, linestyle='-', color=colors[i])
    plt.plot(data_base['BGSD_eff'], linestyle='--', color=colors[i])
plt.legend()
plt.ylabel('$BGSD_{eff}$')
sns.despine()
plt.suptitle(r'$\alpha_{pr}=$' + str(alpha_prot) + r' $\alpha_{lb}=$' + str(alpha_lb) +
             'decision_func=' + decision_function)
savefig(f'BGSD_eff_{decision_function}_alpha_lb{alpha_lb}')

# bar plot with time it takes until all unpriviliged are high prospect
# x axis is scenarios, and we want to color the scenarios the same way as in the other plots
# to differentiate base and full model, we use different hatching.
# this type of grouping is not directly implemented because we have qualitative x-balues
# therefore, we first need to convert the x values to numeric, and then
# label them again
# https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
plt.figure(figsize=figsize)
xvals = np.unique(df_agg['scenario'])
_x = np.arange(len(xvals))
# this approach is only valid if the order of the scenarios is the same for both model types (which
# is the case here if nothing is changed above, but we check it anyway)
assert(np.array_equal(df_agg['scenario'][df_agg['modeltype']=='base'], df_agg['scenario'][df_agg['modeltype']=='full']))
width = 0.4
plt.bar(_x-width/2,
        df_agg['t_all_upriv_highpros'][df_agg['modeltype'] == 'base'],
        width=width, color=colors, hatch='//', label='base')
plt.bar(_x+width/2,
        df_agg['t_all_upriv_highpros'][df_agg['modeltype'] == 'full'],
        width=width, color=colors, hatch='--', label='full')
plt.xticks(_x, xvals)
plt.legend()

plt.suptitle(r'$\alpha_{pr}=$' + str(alpha_prot) + r' $\alpha_{lb}=$' + str(alpha_lb) +
             'decision_func=' + decision_function)
sns.despine()
plt.xlabel('scenario')
plt.ylabel('time until all underpriviliged are high prospect')
plt.tight_layout()
savefig(f't_all_upriv_highpros_{decision_function}_alpha_lb{alpha_lb}')
