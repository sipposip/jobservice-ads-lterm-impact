"""

design note: for computational efficiency, everything is done with numpy/pandas vector operations
pools of people all begin with df_

parameters: T_u : time already unemployed


ideas: we need a warming-up cycle to build up historical data
in this warming-up phase the PES does not do anything

"""

import os

from pylab import plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
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


def assign_jobs(df, loc=0):
    """compute probability of finding a job for each individuals
    and then seperate the input individuals into those that get assigned a job
    and those that remain workless

    return: df_found_job, df_remains_workless
    """
    # in the influx population, skill is normally distributed (var=1)
    # as probability for finding a job, we use the cdf of that distribution
    p = stats.norm(loc=loc, scale=1).cdf(df['s_real'])
    # now select who finds a job. we can do this via drawing from a uniform distribution in [0,1]
    # and then select those where p is larger that that
    idcs_found_job = p > stats.uniform.rvs(size=len(p))
    # we copy the data to avoid potential problems with referencing etc
    df_found_job = df[idcs_found_job].copy()
    df_remains_workless = df[~idcs_found_job].copy()
    assert (len(df_found_job) + len(df_remains_workless) == len(df))
    return df_found_job, df_remains_workless


def train_model(x):
    # T_u = [e[2] for e in x]
    # X = [e[:2] for e in x]
    # classes = (T_u > np.mean(T_u)).astype(int)
    T_u = x['T_u']
    X = x[['x1','x_prot']]
    classes = (T_u > np.mean(T_u)).astype(int)
    return linear_model.LogisticRegression().fit(X, classes)


def update_historical_data(df_hist, df_found_job):
    """the historical data is the record of poeple who have
    found a job eventually"""
    df_hist = pd.concat([df_hist, df_found_job], axis=0)
    return df_hist


# steps of the model
rand_seed = 998654  # fixed random seed for reproducibility
np.random.seed(rand_seed)
n_population = 1000
alpha_prot = 2  # influence of alpha_prot on x2
maxval = 2
n_spinup = 50
T_w_max = 1
# 1 generate initial data
df_active = draw_data_from_influx(n_population, alpha_prot, maxval)
# initialize empty historical data
df_hist = pd.DataFrame()
df_waiting = pd.DataFrame()

for step in range(100):
    print(step)
    df_found_job, df_remains_workless = assign_jobs(df_active)
    n_found_job, n_remains_workless = len(df_found_job), len(df_remains_workless)
    # update historical data
    df_hist = update_historical_data(df_hist, df_found_job)
    # increase T_u of the ones that remained workless
    df_remains_workless['T_u'] = df_remains_workless['T_u'] + 1

    if step > n_spinup:
        model = train_model(df_hist)
        # group the current jobless people into the two groups
        classes = model.predict(df_remains_workless[['x1', 'x_prot']])
        df_highpros = df_remains_workless[classes == 1].copy()
        df_lowpros = df_remains_workless[classes == 0].copy()
        n_highpros, n_lowpros = len(df_highpros), len(df_lowpros)
        assert(len(df_highpros)+len(df_lowpros)==len(df_remains_workless))

        # TODO
        # implement intervention modle here
        # END TOOO

        df_remains_workless = df_highpros
        # for the lowpros group, we need a new attribute that describes how long they are already
        # in the waiting position, which starts at 0
        df_lowpros['T_w'] = 0
        # move the ones that reached the final time in the waiting group to the normal
        # job seeker group
        if len(df_waiting) > 0:
            df_back_idcs = df_waiting['T_w'] == T_w_max
            df_back = df_waiting[df_back_idcs]
            df_waiting = df_waiting[~df_back_idcs]
            if len(df_back) > 0:
                df_back = df_back.drop(columns='T_w')
                df_remains_workless = pd.concat([df_remains_workless, df_back])
            df_waiting['T_w'] = df_waiting['T_w'] + 1

        df_waiting = pd.concat([df_waiting, df_lowpros])
        n_waiting = len(df_waiting)
        print(n_waiting)
    # draw new people from influx to replace the ones that found a job
    df_new = draw_data_from_influx(n_found_job, alpha_prot, maxval)
    df_active = pd.concat([df_remains_workless, df_new], axis=0)
    print(f'active:{len(df_active)} waiting:{len(df_waiting)} ')
    assert(len(df_active)+len(df_waiting)==n_population)

sns.jointplot('x1', 'T_u', data=df_hist)
