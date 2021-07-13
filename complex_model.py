"""

design note: for computational efficiency, everything is done with numpy/pandas vector operations
pools of people all begin with df_

features of individuals:
    x_pr: protected attribute
    x1: first skill feature (correlated with x_pr)
    x2: second skill feature (uncorrelated with x_pr)
    T_u : time already unemployed


model parameters:
    delta_T_u: time an individual spends in the lowprospect waiting group
    ...
    ...

differences to simple model:
in the simple model, prospect on the labour market is a direct (deterministic) function
of s_real, therefore we know the real prospect, and can use it in the definitions
of the scenarios.
in the complex model, the real prospect is a probabilistic function of s_real, and emerges
only throughout the simulation.
Therefore, we actually cannot know how long an individual would need to find a job, even
if we know s_real.
For the scenario, we instead use an estimation of T_u based on a logisitc model using s_real as input,
trained on the historical data.



performance topics:
    the model needs a data structure (the history) that grows with every timestep, but in an unpredictable way.
    This is something that cannot be done efficiently with numpy arrays or pandas dataframes.
    In principle one could use standard python lists for this (which can be grown dynamically with reasonable
    speed and memory requirements), but here this is not an option, because at every timestep we need the history
    as input for the training of the logistic regression, and for this it needs to be an array or dataframe.
    if we were to convert the dynamic list before training at every timestep we would loose the speed increase.
    Therefore, I chose to use dataframes anyway, and partly circumvent the problems by creating
    a dataframe filled with nans with the maximum possible size that can happen in the simulation




"""

import os
from tqdm import trange
from pylab import plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats, special
from sklearn import linear_model, metrics

sns.set_context('notebook', font_scale=1)
sns.set_palette('colorblind')
plotdir = './plots_complexmodel/'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)


def savefig(fname):
    plt.savefig(plotdir + fname + '.pdf')
    plt.savefig(plotdir + fname + '.svg')


def draw_data_from_influx(n, alpha_prot, maxval):
    """
    generate n individuals from the (fixed) background influx distribution
    :return: dataframe
    """
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
    s_real = compute_skill(x1, x2)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x_prot': x_prot, 's_real': s_real})
    # set T_u to 0 for all individuals
    df['T_u'] = 0
    return df


def compute_skill(x1, x2):
    # TODO HACK!!
    s_real = (x1 + x2) / 2
    # s_real = (x1) / 2
    return s_real


def assign_jobs(df, loc, scale):
    """compute probability of finding a job for each individuals
    and then seperate the input individuals into those that get assigned a job
    and those that remain workless
    in the influx population, skill is normally distributed (var=1)
    we make the probability of finding a job a function of s_real
    p = p('s_real'). Note that p('s_real') is NOT a probability density function of s_real, but it is a function
    that returns the probability of the event "finds a job" given a certain value of s_real
    we model it as a logistic function (which is automatically between [0,1])
    loc: location of the logistic probability function
    scale: scale of the logistic probability function
    return: df_found_job, df_remains_workless
    """
    p = special.expit(df['s_real'] * scale - loc)
    # now select who finds a job. we can do this via drawing from a uniform distribution in [0,1]
    # and then select those where p is larger that that
    idcs_found_job = p > stats.uniform.rvs(size=len(p))
    # we copy the data to avoid potential problems with referencing etc
    df_found_job = df[idcs_found_job].copy()
    df_remains_workless = df[~idcs_found_job].copy()
    assert (len(df_found_job) + len(df_remains_workless) == len(df))
    return df_found_job, df_remains_workless


def compute_class(df, class_boundary):
    """compute lowpros (1) and highpros (0) class"""
    return (df['T_u'] < class_boundary).astype(int)


def train_model(df, modeltype, class_boundary):
    """
        train a logistic regression model
    """
    if modeltype == 'full':
        X = df[['x1', 'x_prot']]
    elif modeltype == 'base':
        X = df[['x1']]
    elif modeltype == 'real':
        X = df[['s_real']]
    classes = compute_class(df, class_boundary)
    return linear_model.LogisticRegression().fit(X, classes)


def predict(df, model, modeltype):
    """
        make predictions with trained model
    """
    if modeltype == 'full':
        X = df[['x1', 'x_prot']]
    elif modeltype == 'base':
        X = df[['x1']]
    elif modeltype == 'real':
        X = df[['s_real']]
    return model.predict(X)


def intervention_model(x1, x2, real_class, pred_class, k_matrix):
    """
        update features based on intervention models. works with scalars and vectors.
        The scenario is encoded in the 2x2 array k_matrix
    """
    assert (k_matrix.shape == (2, 2))
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


k_matrix = np.array([[0.1, 0.1],
                     [0.1, 0.1]])

# parameters
rand_seed = 998654  # fixed random seed for reproducibility
np.random.seed(rand_seed)
n_population = 10000
alpha_prot = 2  # influence of alpha_prot on x2
maxval = 2
tsteps = 400  # steps after spinup
n_spinup = 400
n_retain_from_spinup = 200
delta_T_u = 5  # time lowpros are withdrawn from active group
T_u_max = 100  # time after which workless individuals leave the system automatically
class_boundary = 10  # in time-units
jobmarket_function_loc = 0
jobmarket_function_scale = 6
modeltype='full'
# generate initial data
# for person-pools we use dataframes, and we always use "df_" as prefix to make clear
# that something is a pool
df_active = draw_data_from_influx(n_population, alpha_prot, maxval)
# initialize empty data pools data
n_hist_max = n_population * (tsteps + n_retain_from_spinup)
# for the history, we make a dataframe of fixed (la
# rge) datasize because extending dataframes is so slow in python
# we fill it with nans, and then slowly fill it up with data
df_hist = pd.DataFrame(np.nan, index=np.arange(n_hist_max), columns=['x1', 'x2', 'x_prot', 's_real', 'T_u', 'step'])

df_waiting = pd.DataFrame()
n_waiting = len(df_waiting)
model_evolution = []

if n_retain_from_spinup > n_spinup:
    raise ValueError('n_retain_from_spinup  must be larger than n_spinup!')

for step in trange(n_spinup + tsteps):
    # assign jobs
    df_found_job, df_remains_workless = assign_jobs(df_active, jobmarket_function_loc, jobmarket_function_scale)
    n_found_job, n_remains_workless = len(df_found_job), len(df_remains_workless)
    df_found_job['step'] = step
    # update historical data
    # find first free row and append from there on
    start_idx = np.argmax(df_hist['x1'].isna())
    df_hist.iloc[start_idx:start_idx + len(df_found_job)] = df_found_job
    # increase T_u of the ones that remained workless
    df_remains_workless['T_u'] = df_remains_workless['T_u'] + 1

    # at end of spinup, crop the history
    if step == n_spinup - 1:
        df_hist_start = df_hist.copy()
        model_evolution_start = model_evolution[:]
        df_hist = df_hist.iloc[np.argmax(df_hist['step'] == n_spinup - n_retain_from_spinup):]
        model_evolution = model_evolution[-n_retain_from_spinup:]

    # remove individuals with T_u > T_u_max
    idx_remove = df_remains_workless['T_u'] > T_u_max
    n_removed = sum(idx_remove)  # idx_remove is a boolean index, so sum gives the number of Trues
    n_removed = sum(idx_remove)  # idx_remove is a boolean index, so sum gives the number of Trues
    df_remains_workless = df_remains_workless[~idx_remove]
    if step > n_spinup:
        # train model on all accumulated historical data
        # for this we need to extract the part of df_hist that is already filled with data
        df_hist_nonan = df_hist.iloc[:np.argmax(df_hist.x1.isna())]
        # our standard model is the
        model = train_model(df_hist_nonan, modeltype, class_boundary)
        model_real = train_model(df_hist_nonan, 'real', class_boundary)

        # evaluate on historical training data
        classes_true_hist = compute_class(df_hist_nonan, class_boundary)
        classes_pred_hist = predict(df_hist_nonan, model, modeltype)
        frac_pred_highpros_hist = np.mean(classes_pred_hist == 0)
        frac_pred_lowpros_hist = np.mean(classes_pred_hist == 1)
        frac_true_highpros_hist = np.mean(classes_true_hist == 0)
        frac_true_lowpros_hist = np.mean(classes_true_hist == 1)
        # compute metrics on the predictions
        accur = metrics.accuracy_score(classes_true_hist, classes_pred_hist)
        recall = metrics.recall_score(classes_true_hist, classes_pred_hist)
        precision = metrics.precision_score(classes_true_hist, classes_pred_hist)

        # group the current jobless people into the two groups
        classes_pred = predict(df_remains_workless, model, modeltype)
        classes_true = predict(df_remains_workless, model_real, 'real')
        frac_highpros = np.mean(classes_pred)
        df_upd = df_remains_workless
        df_upd['x1'], df_upd['x2'] = intervention_model(df_upd['x1'], df_upd['x2'], classes_true,
                                                        classes_pred, k_matrix)

        df_highpros = df_upd[classes_pred == 1].copy()
        df_lowpros = df_upd[classes_pred == 0].copy()
        n_highpros, n_lowpros = len(df_highpros), len(df_lowpros)
        assert (len(df_highpros) + len(df_lowpros) == len(df_upd))

        if delta_T_u > 0:
            # for the lowpros group, we need a new attribute that describes how long they are already
            # in the waiting position, which starts at 0
            df_lowpros['T_w'] = 0

            # only the highpros are retained, they will be complemented by the ones from
            # the waiting pool and by new ones later on
            df_remains_workless = df_highpros
            # move the ones that reached the final time in the waiting group to the normal
            # job seeker group
            if n_waiting > 0:
                df_back_idcs = df_waiting['T_w'] == delta_T_u
                df_back = df_waiting[df_back_idcs]
                df_waiting = df_waiting[~df_back_idcs]
                if len(df_back) > 0:
                    df_back = df_back.drop(columns='T_w')
                    df_back['T_u'] += delta_T_u
                    df_remains_workless = pd.concat([df_remains_workless, df_back])
                df_waiting['T_w'] = df_waiting['T_w'] + 1

                assert (np.all(df_waiting['T_w'] <= delta_T_u))
            # add the new lowpros to the waiting group
            df_waiting = pd.concat([df_waiting, df_lowpros])
            n_waiting = len(df_waiting)
        else:
            # in case there is no waiting time for the lowpros, all go together in the
            # group with workless
            df_remains_workless = pd.concat([df_highpros, df_lowpros])
    else:
        # in the spinup phase we dont have predictions and therefore no prediction performance metrics
        # set values to be used in the record during the spinup pahse
        accur = np.nan
        precision = np.nan
        recall = np.nan
        frac_pred_highpros_hist = np.nan
        frac_pred_lowpros_hist = np.nan
        frac_true_highpros_hist = np.nan
        frac_true_lowpros_hist = np.nan
        frac_highpros = np.nan

    # draw new people from influx to replace the ones that found a job and add them
    # to the pool of active jobseekers
    df_new = draw_data_from_influx(n_found_job + n_removed, alpha_prot, maxval)
    df_active = pd.concat([df_remains_workless, df_new], axis=0)
    n_active = len(df_active)
    assert (n_active + n_waiting == n_population)
    df_active_and_waiting = pd.concat([df_active, df_waiting], axis=0)
    # compute summary statistics
    model_evolution.append(pd.DataFrame({
        'n_active': n_active,
        'n_waiting': n_waiting,
        'n_found_jobs': n_found_job,
        'accuracy': accur,
        'recall': recall,
        'precision': precision,
        's_all': np.mean(df_active_and_waiting['s_real']),
        's_priv': np.mean(df_active_and_waiting['s_real'][df_active_and_waiting['x_prot'] == 1]),
        's_upriv': np.mean(df_active_and_waiting['s_real'][df_active_and_waiting['x_prot'] == 0]),
        'frac_pred_highpros_hist': frac_pred_highpros_hist,
        'frac_pred_lowpros_hist': frac_pred_lowpros_hist,
        'frac_true_highpros_hist': frac_true_highpros_hist,
        'frac_true_lowpros_hist': frac_true_lowpros_hist,
        'frac_highpros': frac_highpros,
    }, index=[step]))

model_evolution = pd.concat(model_evolution)
# remove tha tail with NaNs
df_hist = df_hist.dropna()

# ---- plot the results ----

# plot jobmarket probability function
plt.figure()
x = np.linspace(-2, 2, 100)
plt.plot(x, special.expit(x * jobmarket_function_scale - jobmarket_function_loc))
plt.xlabel('s_real')
plt.ylabel('P finding a job')
sns.despine()

# plot distribution of T_u at the end of the history (so the
# distribution of T_u of the individuals who found a job a the last timestep)
df_hist_end_of_spinup = df_hist[df_hist['step'].between(n_spinup - 10, n_spinup)]
sns.displot(df_hist_end_of_spinup['T_u'])
sns.jointplot('x1', 'T_u', data=df_hist_end_of_spinup)
# plot the relation ov x1 and T_u over the whole period
sns.jointplot('x1', 'T_u', data=df_hist)

df_hist_last = df_hist[df_hist['step'] == df_hist['step'].max()]
classes_true_hist_last = classes_true_hist[df_hist['step'] == df_hist['step'].max()]
classes_pred_hist_last = classes_pred_hist[df_hist['step'] == df_hist['step'].max()]
sns.jointplot('x1', 'T_u', data=df_hist_last, hue=classes_true_hist_last)
sns.jointplot('x1', 'T_u', data=df_hist_last, hue=classes_pred_hist_last)
sns.jointplot('x1', 'T_u', data=df_hist_last, hue='x_prot')

mean_Tu_priv = df_hist_last[df_hist_last['x_prot'] == 1]['T_u'].mean()
mean_Tu_unpriv = df_hist_last[df_hist_last['x_prot'] == 0]['T_u'].mean()
plt.figure()
sns.histplot(data=df_hist_last, x='T_u', hue='x_prot')
plt.title(f'mean: unpriv (x_prot=0):{mean_Tu_unpriv:.2f}, priv (x_prot=0):{mean_Tu_priv:.2f}')

tmean_until_job = df_hist.groupby('step')['T_u'].mean()
tmean_until_job_priv = df_hist[df_hist['x_prot'] == 1].groupby('step')['T_u'].mean()
tmean_until_job_upriv = df_hist[df_hist['x_prot'] == 0].groupby('step')['T_u'].mean()
tmean_until_job_cumulative = np.cumsum(tmean_until_job) / np.arange(len(tmean_until_job))

# plot modeltime vs different metrics/measures
n_plots = 6
plt.figure(figsize=(6.4, 11))
ax1 = plt.subplot(n_plots, 1, 1)
plt.plot(tmean_until_job, label='all')
plt.plot(tmean_until_job_priv, label='priv')
plt.plot(tmean_until_job_upriv, label='upriv')
plt.legend()
plt.ylabel('T_u')
sns.despine()
ax = plt.subplot(n_plots, 1, 2)
model_evolution[['accuracy', 'recall', 'precision']].plot(ax=ax)
# since accuracy, recall and precision start with NANs, the plotting
# function omits them and starts the xaxis only at the first valid value
# therfore we have to set the xlims
plt.xlim(*ax1.get_xlim())
sns.despine()
ax = plt.subplot(n_plots, 1, 3)
model_evolution[['s_all', 's_priv', 's_upriv']].plot(ax=ax)
sns.despine()
plt.ylabel('$s_{real}$')
ax = plt.subplot(n_plots, 1, 4)
plt.plot(tmean_until_job_cumulative)
sns.despine()
plt.ylabel('cumulative T_u')

ax = plt.subplot(n_plots, 1, 5)
model_evolution[['frac_highpros', 'frac_true_highpros_hist', 'frac_true_lowpros_hist',
                 'frac_pred_highpros_hist', 'frac_pred_lowpros_hist']].plot(ax=ax)
sns.despine()

ax = plt.subplot(n_plots, 1, 6)
model_evolution[['n_waiting', 'n_found_jobs']].plot(ax=ax)
sns.despine()
plt.xlabel('t')

plt.tight_layout()
