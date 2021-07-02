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


TODO: implement intervention model and scenarios (can basically be copied from simple model)

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
        x1 = stats.norm.rvs(-maxval, maxval, size=n)
        x2 = 1 / 2 * (alpha_prot * (x_prot - 0.5) + stats.truncnorm.rvs(-maxval, maxval, size=n))
    s_real = compute_skill(x1, x2)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x_prot': x_prot, 's_real': s_real})
    # set T_u to 0 for all individuals
    df['T_u'] = 0
    return df


def compute_skill(x1, x2):
    s_real = (x1 + x2) / 2
    return s_real


def assign_jobs(df, loc, scale):
    """compute probability of finding a job for each individuals
    and then seperate the input individuals into those that get assigned a job
    and those that remain workless

    return: df_found_job, df_remains_workless
    """
    # in the influx population, skill is normally distributed (var=1)
    # we make the probability of finding a job a function of s_real
    # p = p('s_real'). Note that p('s_real') is NOT a probability density function of s_real, but it is a function
    # that returns the probability of the event "finds a job" given a certain value of s_real
    # we model it as a logistic function (which is automatically between [0,1])
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
    """compute lowpros (0) and highpros (1) class"""
    return (df['T_u'] > class_boundary).astype(int)


def train_model(df, modeltype, class_boundary):
    """
        train a logistic regression model
        for now, the classes are defined as below and above mean T_u in the training data
    """
    if modeltype == 'full':
        X = df[['x1', 'x_prot']]
    elif modeltype == 'base':
        X = df[['x1']]
    classes = compute_class(df, class_boundary)
    return linear_model.LogisticRegression().fit(X, classes)


def predict(df, model, modeltype):
    """
        train a logistic regression model
        for now, the classes are defined as below and above mean T_u in the training data
    """
    if modeltype == 'full':
        X = df[['x1', 'x_prot']]
    elif modeltype == 'base':
        X = df[['x1']]
    return model.predict(X)


# parameters
rand_seed = 998654  # fixed random seed for reproducibility
np.random.seed(rand_seed)
n_population = 10000
alpha_prot = 2  # influence of alpha_prot on x2
maxval = 2
tsteps = 400  # steps after spinup
n_spinup = 400
n_retain_from_spinup = 200
delta_T_u = 10   # time lowpros are withdrawn from active group
T_u_max = 100 # time after which workless individuals leave the system automatically
modeltype = 'full'
class_boundary = 40  # in time-units
jobmarket_function_loc = 0
jobmarket_function_scale = 10
# generate initial data
# for person-pools we use dataframes, and we always use "df_" as prefix to make clear
# that something is a pool
df_active = draw_data_from_influx(n_population, alpha_prot, maxval)
# initialize empty data pools data
n_hist_max = n_population * (tsteps + n_retain_from_spinup)
# for the history, we make a dataframe of fixed (large) datasize because extending dataframes is so slow in python
# we fill it with nans, and then slowly fill it up with data
df_hist = pd.DataFrame(np.nan, index=np.arange(n_hist_max), columns=['x1', 'x2', 'x_prot', 's_real', 'T_u', 'step'])

df_waiting = pd.DataFrame()
n_waiting = len(df_waiting)
model_evolution = []

if n_retain_from_spinup > n_spinup:
    raise ValueError('crop_history_spinup  must be larger than n_spinup!')

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
    df_remains_workless = df_remains_workless[~idx_remove]
    if step > n_spinup:
        # train model on all accumulated historical data
        df_hist_nonan = df_hist.iloc[:np.argmax(df_hist.x1.isna())]
        model = train_model(df_hist_nonan, modeltype, class_boundary)
        # group the current jobless people into the two groups
        classes = predict(df_remains_workless, model, modeltype)
        classes_true = compute_class(df_remains_workless, class_boundary)
        accur = metrics.accuracy_score(classes_true, classes)
        recall = metrics.recall_score(classes_true, classes)
        precision = metrics.precision_score(classes_true, classes)
        # here we deviate from the terminology used for the simple model.
        # since the ml-model is based on predicting the unemployment time, class 1 indicates
        # the low-prospect group (long expected unemployment time)
        df_highpros = df_remains_workless[classes == 0].copy()
        df_lowpros = df_remains_workless[classes == 1].copy()
        n_highpros, n_lowpros = len(df_highpros), len(df_lowpros)
        assert (len(df_highpros) + len(df_lowpros) == len(df_remains_workless))

        # TODO
        # implement intervention model here
        # END TOOO


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

            # add the new lowprps to the waiting group
            df_waiting = pd.concat([df_waiting, df_lowpros])
            n_waiting = len(df_waiting)
        else:
            # in case there is no waiting time for the lowpros, all go together in the
            # gorup with workless
            df_remains_workless = pd.concat([df_highpros, df_lowpros])
    else:
        # set values to be used in the record during the spinup pahse
        accur = np.nan
        precision = np.nan
        recall = np.nan
    # draw new people from influx to replace the ones that found a job
    df_new = draw_data_from_influx(n_found_job + n_removed, alpha_prot, maxval)
    df_active = pd.concat([df_remains_workless, df_new], axis=0)
    n_active = len(df_active)
    assert (n_active + n_waiting == n_population)
    model_evolution.append(pd.DataFrame({
        'n_active': n_active,
        'n_waiting': n_waiting,
        'n_found_jobs': n_found_job,
        'accuracy': accur,
        'recall': recall,
        'precision': precision
    }, index=[step]))

model_evolution = pd.concat(model_evolution)
# remove tha tail with NaNs
df_hist = df_hist.dropna()
# plot jobmarket probability function
plt.figure()
x = np.linspace(-3, 3, 100)
plt.plot(x, special.expit(x * jobmarket_function_scale - jobmarket_function_loc))
plt.xlabel('s_real')
plt.ylabel('P finding a job')

sns.jointplot('x1', 'T_u', data=df_hist)

model_evolution.plot()

plt.figure()
plt.subplot(311)
sns.boxplot(x='step', y='T_u', data=df_hist, fliersize=1)
plt.subplot(312)
sns.boxplot(x='step', y='T_u', data=df_hist, showfliers=False)
plt.subplot(313)
sns.lineplot(x='step', y='T_u', data=df_hist, ci=None)

model_evolution[['accuracy', 'recall', 'precision']].plot()
plt.ylabel('accuracy')

tmean_until_job = df_hist.groupby('step')['T_u'].mean()
tmean_until_job_cumulative = np.cumsum(tmean_until_job) / np.arange(len(tmean_until_job))

plt.figure()
plt.subplot(211)
plt.plot(tmean_until_job)
sns.despine()
plt.ylabel('mean T_u at this step')
plt.subplot(212)
plt.plot(tmean_until_job_cumulative)
sns.despine()
plt.ylabel('cumulative mean T_u')
plt.xlabel('t')

df_hist_end_of_spinup = df_hist[df_hist['step'].between(n_spinup - 10, n_spinup)]
sns.displot(df_hist_end_of_spinup['T_u'])
