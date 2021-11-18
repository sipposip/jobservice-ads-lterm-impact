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

for reproduction: this script needs to be run 2 times, one time with modeltype='full', one time with modeltype='base'

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
import sys
from tqdm import trange
from pylab import plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats, special
from sklearn import linear_model, metrics
import matplotlib.backends.backend_pdf

sns.set_context('notebook', font_scale=1)
sns.set_palette('colorblind')
plotdir = './plots_complexmodel/'
datadir = './data_complexmodel/'
for dir in (plotdir, datadir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
plt.ioff()


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
        x2 = 1 / 2 * (alpha_prot * (x_prot - 0.5) + stats.norm.rvs(size=n))
    s_real = compute_skill(x1, x2)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x_prot': x_prot, 's_real': s_real})
    # set T_u to 0 for all individuals
    df['T_u'] = 0
    return df


def compute_skill(x1, x2):
    s_real = (x1 + x2) / 2
    return s_real


def assign_jobs(df, loc, scale, bias):
    """compute probability of finding a job for each individual
    and then separate the input individuals into those that get assigned a job
    and those that remain workless
    in the influx population, skill is normally distributed (var=1)
    we make the probability of finding a job a function of s_real
    p = p('s_real'). Note that p('s_real') is NOT a probability density function of s_real, but it is a function
    that returns the probability of the event "finds a job" given a certain value of s_real.
    we model it as a logistic function (which is automatically between [0,1])
    loc: location of the logistic probability function
    scale: scale of the logistic probability function
    return: df_found_job, df_remains_workless
    """
    # for a biased labormarket, the location is dependent on the protected attribute
    loc_vec = loc * np.ones(len(df))
    loc_vec = loc_vec - bias * (df[
                                    'x_prot'] - 0.5)  # we need to substract the bias to get  higher probabilities already for lower s_real
    p = special.expit(df['s_real'] * scale - loc_vec)
    # now select who finds a job. we can do this via drawing from a uniform distribution in [0,1]
    # and then select those where p is larger that that
    idcs_found_job = p > stats.uniform.rvs(size=len(p))
    # we copy the data to avoid potential problems with referencing etc
    df_found_job = df[idcs_found_job].copy()
    df_remains_workless = df[~idcs_found_job].copy()
    assert (len(df_found_job) + len(df_remains_workless) == len(df))
    return df_found_job, df_remains_workless


def compute_class(df, class_boundary):
    """compute lowpros (0) or highpros (1) class"""
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
    k_matrix = k_matrix * scale_factor

    x1_max = 2
    x2_max = 2
    # we have four parameters, for four cases (real class can be 1 or 2, and real class can be 1 or two)
    # make an array with k values for the correct real_class - pred_class combination, then we can use
    # vector operations and avoid a slow for loop
    kvec = np.zeros_like(x1)
    kvec[(real_class == 0) & (pred_class == 0)] = k_matrix[0, 0]
    kvec[(real_class == 0) & (pred_class == 1)] = k_matrix[0, 1]
    kvec[(real_class == 1) & (pred_class == 0)] = k_matrix[1, 0]
    kvec[(real_class == 1) & (pred_class == 1)] = k_matrix[1, 1]

    x1_highpro = x1[pred_class == 1]
    x2_highpro = x2[pred_class == 1]
    x1_lowpro = x1[pred_class == 0]
    x2_lowpro = x2[pred_class == 0]

    x1_new = np.zeros_like(x1)
    x2_new = np.zeros_like(x2)

    x1_new[pred_class == 1] = np.maximum(x1_highpro + (x1_max - x1_highpro) * kvec[pred_class == 1], x1_highpro)
    x2_new[pred_class == 1] = np.maximum(x2_highpro + (x2_max - x2_highpro) * kvec[pred_class == 1], x2_highpro)

    # we have to treat highpro and lowpro (predicted) seperately here in order to adapt
    # for the fact that the ones in the waiting group are updated only once
    for _ in range(delta_T_u + 1):
        x1_lowpro = np.maximum(x1_lowpro + (x1_max - x1_lowpro) * kvec[pred_class == 0], x1_lowpro)
        x2_lowpro = np.maximum(x2_lowpro + (x2_max - x2_lowpro) * kvec[pred_class == 0], x2_lowpro)

    x1_new[pred_class == 0] = x1_lowpro
    x2_new[pred_class == 0] = x2_lowpro
    x1_new[pred_class == 1] = x1_highpro
    x2_new[pred_class == 1] = x2_highpro

    return x1_new, x2_new


configs = [
    {'scenario': "onlylow",
     'description': 'only on lowprospect, no class-dependent effect',
     'k_matrix': np.array([[1, 0],
                           [1, 0]]),
     },
    {'scenario': "onlyhigh",
     'description': 'only on highprospect, no class-dependent effect',
     'k_matrix': np.array([[0, 1],
                           [0, 1]]),
     },
    {'scenario': "balanced",
     'description': 'no targeting, no class-dependent effect',
     'k_matrix': np.array([[1, 1],
                           [1, 1]]),
     },
    {'scenario': "balanced_errors_penalized",
     'description': 'no targeting, no class-dependent effect',
     'k_matrix': np.array([[1, 1 / 2],
                           [1 / 2, 1]]),
     },

]

# parameters
rand_seed = 998654  # fixed random seed for reproducibility
scale_factor = 1 / 500
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
labormarket_bias = 2  # bias to jobmarket_scale. ran with 2, todo but this is much too high!!
modeltype = 'full'  # full | base
scenario = sys.argv[1]


for config in configs:
    scenario = config['scenario']

    k_matrix = config['k_matrix']

    # parameterstring string for filenames
    paramstr = '_'.join(
        [str(e) for e in (alpha_prot, tsteps, n_spinup, n_retain_from_spinup, delta_T_u, T_u_max, class_boundary,
                          jobmarket_function_loc, jobmarket_function_scale, scale_factor, modeltype,
                          labormarket_bias, scenario)])

    # generate initial data
    # for person-pools we use dataframes, and we always use "df_" as prefix to make clear
    # that something is a pool
    df_active = draw_data_from_influx(n_population, alpha_prot, maxval)
    # initialize empty data pools data
    # for the history, we make a dataframe of fixed (la
    # rge) datasize because extending dataframes is so slow in python
    # we fill it with nans, and then step for step fill it up with data
    n_hist_max = n_population * (tsteps + n_retain_from_spinup)
    df_hist = pd.DataFrame(np.nan, index=np.arange(n_hist_max), columns=['x1', 'x2', 'x_prot', 's_real', 'T_u', 'step'])

    # initialize df_waiting with correct keys
    df_waiting = pd.DataFrame(columns=df_active.keys())
    n_waiting = len(df_waiting)
    model_evolution = []

    if n_retain_from_spinup > n_spinup:
        raise ValueError('n_retain_from_spinup  must be larger than n_spinup!')

    for step in trange(n_spinup + tsteps):
        # assign jobs
        df_found_job, df_remains_workless = assign_jobs(df_active, jobmarket_function_loc, jobmarket_function_scale,
                                                        labormarket_bias)
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
            # for this we need to extract the part of df_hist that is already filled with data
            df_hist_nonan = df_hist.iloc[:np.argmax(df_hist.x1.isna())]
            # our standard model is the
            model = train_model(df_hist_nonan, modeltype, class_boundary)
            model_real = train_model(df_hist_nonan, 'real', class_boundary)

            # store the model weight for the protected attribute. this is only available for the 'full' model,
            # for the 'base' model we set it to zero
            if modeltype == 'full':
                # the coef_ array is 2d, 1st dimension is empty, x_prot is the 2nd element along the 2nd dimension
                coef2 = model.coef_[0, 1]
            elif modeltype == 'base':
                coef2 = 0
            else:
                raise Exception('should never get here')
            coef1 = model.coef_[0, 0]

            # evaluate on historical training data
            classes_true_hist = compute_class(df_hist_nonan, class_boundary)
            classes_pred_hist = predict(df_hist_nonan, model, modeltype)
            frac_pred_highpros_hist = np.mean(classes_pred_hist == 1)
            frac_pred_lowpros_hist = np.mean(classes_pred_hist == 0)
            frac_true_highpros_hist = np.mean(classes_true_hist == 1)
            frac_true_lowpros_hist = np.mean(classes_true_hist == 0)
            # compute metrics on the predictions
            accur = metrics.accuracy_score(classes_true_hist, classes_pred_hist)
            recall = metrics.recall_score(classes_true_hist, classes_pred_hist)
            precision = metrics.precision_score(classes_true_hist, classes_pred_hist)
            # same metrics, split up by protected group
            idx_priv = df_hist_nonan['x_prot'] == 1
            accur_priv = metrics.accuracy_score(classes_true_hist[idx_priv], classes_pred_hist[idx_priv])
            recall_priv = metrics.recall_score(classes_true_hist[idx_priv], classes_pred_hist[idx_priv])
            precision_priv = metrics.precision_score(classes_true_hist[idx_priv], classes_pred_hist[idx_priv])
            idx_upriv = df_hist_nonan['x_prot'] == 0
            accur_upriv = metrics.accuracy_score(classes_true_hist[idx_upriv], classes_pred_hist[idx_upriv])
            recall_upriv = metrics.recall_score(classes_true_hist[idx_upriv], classes_pred_hist[idx_upriv])
            precision_upriv = metrics.precision_score(classes_true_hist[idx_upriv], classes_pred_hist[idx_upriv])

            # group the current jobless people into the two groups
            classes_pred = predict(df_remains_workless, model, modeltype)
            classes_true = predict(df_remains_workless, model_real, 'real')
            frac_highpros_pred = np.mean(classes_pred)
            frac_highpros_true = np.mean(classes_true)
            frac_upriv_in_highpros = len(df_remains_workless[classes_pred == 1].query('x_prot==0')) / len(
                df_remains_workless.query('x_prot==0'))
            df_upd = df_remains_workless
            df_upd['x1'], df_upd['x2'] = intervention_model(df_upd['x1'], df_upd['x2'], classes_true,
                                                            classes_pred, k_matrix)
            # since we updated x1 and x2, we also need to update s_real
            df_upd['s_real'] = compute_skill(df_upd['x1'], df_upd['x2'])
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
                else:
                    df_waiting = df_lowpros
                n_waiting = len(df_waiting)
            else:
                # in case there is no waiting time for the lowpros, all go together in the
                # group with workless
                df_remains_workless = pd.concat([df_highpros, df_lowpros])
        else:
            # in the spinup phase we dont have predictions and therefore no prediction performance metrics
            # we set values to be used in the record during the spinup phase to nan
            accur = np.nan
            precision = np.nan
            recall = np.nan
            accur_priv = np.nan
            precision_priv = np.nan
            recall_priv = np.nan
            accur_upriv = np.nan
            precision_upriv = np.nan
            recall_upriv = np.nan
            frac_pred_highpros_hist = np.nan
            frac_pred_lowpros_hist = np.nan
            frac_true_highpros_hist = np.nan
            frac_true_lowpros_hist = np.nan
            frac_highpros_pred = np.nan
            frac_highpros_true = np.nan
            frac_upriv_in_highpros = np.nan
            coef1 = np.nan
            coef2 = np.nan

        # draw new people from influx to replace the ones that found a job and add them
        # to the pool of active jobseekers
        df_new = draw_data_from_influx(n_found_job + n_removed, alpha_prot, maxval)
        df_active = pd.concat([df_remains_workless, df_new], axis=0)
        n_active = len(df_active)
        assert (n_active + n_waiting == n_population)
        df_active_and_waiting = pd.concat([df_active, df_waiting], axis=0)

        # compute summary statistics
        _df = pd.DataFrame({
            'n_active': n_active,
            'n_active_priv': np.sum(df_active_and_waiting['x_prot'] == 1),
            'n_active_upriv': np.sum(df_active_and_waiting['x_prot'] == 0),
            'n_waiting': n_waiting,
            'n_waiting_priv': np.sum(df_waiting['x_prot'] == 1),
            'n_waiting_upriv': np.sum(df_waiting['x_prot'] == 0),
            'n_found_jobs': n_found_job,
            'n_found_jobs_priv': np.sum(df_found_job['x_prot'] == 1),
            'n_found_jobs_upriv': np.sum(df_found_job['x_prot'] == 0),
            'Tu_found_jobs': np.mean(df_found_job['T_u']),
            'Tu_found_jobs_upriv': np.mean(df_found_job[df_found_job['x_prot'] == 0]['T_u']),
            'Tu_found_jobs_priv': np.mean(df_found_job[df_found_job['x_prot'] == 1]['T_u']),
            'accuracy': accur,
            'recall': recall,
            'precision': precision,
            'accuracy_priv': accur_priv,
            'recall_priv': recall_priv,
            'precision_priv': precision_priv,
            'accuracy_upriv': accur_upriv,
            'recall_upriv': recall_upriv,
            'precision_upriv': precision_upriv,
            's_all': np.mean(df_active_and_waiting['s_real']),
            's_priv': np.mean(df_active_and_waiting['s_real'][df_active_and_waiting['x_prot'] == 1]),
            's_upriv': np.mean(df_active_and_waiting['s_real'][df_active_and_waiting['x_prot'] == 0]),
            'frac_unpriv': np.mean(df_active_and_waiting['x_prot'] == 0),
            'frac_pred_highpros_hist': frac_pred_highpros_hist,
            'frac_pred_lowpros_hist': frac_pred_lowpros_hist,
            'frac_true_highpros_hist': frac_true_highpros_hist,
            'frac_true_lowpros_hist': frac_true_lowpros_hist,
            'frac_highpros_pred': frac_highpros_pred,
            'frac_highpros_true': frac_highpros_true,
            'mean_Tu_current': np.mean(df_active_and_waiting['T_u']),
            'mean_Tu_priv_current': np.mean(df_active_and_waiting['T_u'][df_active_and_waiting['x_prot'] == 1]),
            'mean_Tu_upriv_current': np.mean(df_active_and_waiting['T_u'][df_active_and_waiting['x_prot'] == 0]),
            'frac_upriv_in_highpros': frac_upriv_in_highpros,
        }, index=[step])
        _df['BGSD'] = _df['s_priv'] - _df['s_upriv']
        _df['BGTuD_current'] = _df['mean_Tu_priv_current'] - _df['mean_Tu_upriv_current']
        _df['BGaccuracyD'] = _df['accuracy_priv'] - _df['accuracy_upriv']
        _df['BGprecisionD'] = _df['precision_priv'] - _df['precision_upriv']
        _df['BGrecallD'] = _df['recall_priv'] - _df['recall_upriv']
        _df['inbal_waiting'] = (
                _df['n_waiting_upriv'] / _df['n_active_upriv'] - _df['n_waiting_priv'] / _df['n_active_priv'])

        _df['coef1'] = coef1
        _df['coef2'] = coef2

        model_evolution.append(_df)

    model_evolution = pd.concat(model_evolution)
    # remove tha tail with NaNs
    df_hist = df_hist.dropna()

    # ---- plot the results ---
    # # we store the figures in a list so that we can save them to a single pdf at the end-
    figs = []

    # plot jobmarket probability function
    fig = plt.figure()
    figs.append(fig)
    x = np.linspace(-2, 2, 100)
    plt.plot(x, special.expit(x * jobmarket_function_scale - jobmarket_function_loc))
    plt.xlabel('s_real')
    plt.ylabel('P finding a job')
    sns.despine()

    # plot distribution of T_u at the end of the history (so the
    # distribution of T_u of the individuals who found a job a the last timestep)
    df_hist_end_of_spinup = df_hist[df_hist['step'].between(n_spinup - 10, n_spinup)]
    # some seaborn plotting function create there own figure, which can be accessed
    # with the .fig attribute
    figgrid = sns.displot(df_hist_end_of_spinup['T_u'])
    plt.title('T_u at end of spinup')
    figs.append(figgrid.fig)

    figgrid = sns.jointplot('x1', 'T_u', data=df_hist_end_of_spinup)
    plt.title('end of spinup')
    figs.append(figgrid.fig)

    # plot the relation ov x1 and T_u over the whole period
    figgrid = sns.jointplot('x1', 'T_u', data=df_hist)
    plt.title('whole history')
    figs.append(figgrid.fig)

    df_hist_last = df_hist[df_hist['step'] == df_hist['step'].max()]
    classes_true_hist_last = classes_true_hist[df_hist['step'] == df_hist['step'].max()]
    classes_pred_hist_last = classes_pred_hist[df_hist['step'] == df_hist['step'].max()]
    classes_true_hist_last.name = 'classes_true'
    figgrid = sns.jointplot('x1', 'T_u', data=df_hist_last, hue=classes_true_hist_last)
    plt.suptitle('last timestep')
    figs.append(figgrid.fig)
    figgrid = sns.jointplot('x1', 'T_u', data=df_hist_last, hue=classes_pred_hist_last)
    plt.suptitle('last timestep, hue classes_pred')
    figs.append(figgrid.fig)
    figgrid = sns.jointplot('x1', 'T_u', data=df_hist_last, hue='x_prot')
    plt.suptitle('last timestep')
    figs.append(figgrid.fig)

    mean_Tu_priv = df_hist_last[df_hist_last['x_prot'] == 1]['T_u'].mean()
    mean_Tu_unpriv = df_hist_last[df_hist_last['x_prot'] == 0]['T_u'].mean()
    fig = plt.figure()
    figs.append(fig)
    sns.histplot(data=df_hist_last, x='T_u', hue='x_prot')
    plt.title(f'mean: unpriv (x_prot=0):{mean_Tu_unpriv:.2f}, priv (x_prot=0):{mean_Tu_priv:.2f}')

    tmean_until_job = df_hist.groupby('step')['T_u'].mean()
    tmean_until_job_priv = df_hist[df_hist['x_prot'] == 1].groupby('step')['T_u'].mean()
    tmean_until_job_upriv = df_hist[df_hist['x_prot'] == 0].groupby('step')['T_u'].mean()
    tmean_until_job_cumulative = np.cumsum(tmean_until_job) / np.arange(len(tmean_until_job))

    # plot modeltime vs different metrics/measures
    n_rows = 6
    n_cols = 2
    fig = plt.figure(figsize=(13, 11))
    figs.append(fig)
    ax1 = plt.subplot(n_rows, n_cols, 1)
    plt.plot(tmean_until_job, label='all')
    plt.plot(tmean_until_job_priv, label='priv')
    plt.plot(tmean_until_job_upriv, label='upriv')
    plt.legend()
    plt.ylabel('T_u')
    sns.despine()
    ax = plt.subplot(n_rows, n_cols, 2)
    model_evolution[['accuracy', 'recall', 'precision']].plot(ax=ax)
    # since accuracy, recall and precision start with NANs, the plotting
    # function omits them and starts the xaxis only at the first valid value
    # therfore we have to set the xlims
    plt.xlim(*ax1.get_xlim())
    sns.despine()
    ax = plt.subplot(n_rows, n_cols, 3)
    model_evolution[['s_all', 's_priv', 's_upriv']].plot(ax=ax)
    sns.despine()
    plt.ylabel('$s_{real}$')
    ax = plt.subplot(n_rows, n_cols, 4)
    plt.plot(tmean_until_job_cumulative)
    sns.despine()
    plt.ylabel('cumulative T_u')

    ax = plt.subplot(n_rows, n_cols, 5)
    model_evolution[['frac_highpros_pred', 'frac_highpros_true', 'frac_true_highpros_hist', 'frac_true_lowpros_hist',
                     'frac_pred_highpros_hist', 'frac_pred_lowpros_hist']].plot(ax=ax)
    sns.despine()
    ax = plt.subplot(n_rows, n_cols, 6)
    model_evolution[['n_waiting', 'n_found_jobs', 'n_waiting_upriv', 'n_found_jobs_upriv',
                     'n_waiting_priv', 'n_found_jobs_priv']].plot(ax=ax)
    sns.despine()

    ax = plt.subplot(n_rows, n_cols, 7)
    model_evolution[['mean_Tu_priv_current', 'mean_Tu_upriv_current']].plot(ax=ax)
    sns.despine()

    ax = plt.subplot(n_rows, n_cols, 8)
    model_evolution['BGSD'].plot(ax=ax)
    plt.ylabel('BGSD')
    sns.despine()

    ax = plt.subplot(n_rows, n_cols, 9)
    model_evolution['BGTuD_current'].plot(ax=ax)
    plt.ylabel('BGTuD_current')
    sns.despine()

    ax = plt.subplot(n_rows, n_cols, 10)
    model_evolution[['BGaccuracyD', 'BGprecisionD', 'BGrecallD']].plot(ax=ax)
    plt.xlim(*ax1.get_xlim())
    sns.despine()

    ax = plt.subplot(n_rows, n_cols, 11)
    model_evolution['inbal_waiting'].plot(ax=ax)
    plt.ylabel('inbal_waiting')
    sns.despine()
    plt.xlabel('t')

    ax = plt.subplot(n_rows, n_cols, 12)
    model_evolution[['coef1', 'coef2']].plot(ax=ax)
    plt.ylabel('regression coefficients')
    sns.despine()
    plt.xlabel('t')

    plt.tight_layout()
    savefig(f'complex_model_mainres_{paramstr}')

    # save all figures into a single pdf (one figure per page)
    pdf = matplotlib.backends.backend_pdf.PdfPages(f'{plotdir}/complex_model_allplots_{paramstr}.pdf')
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()

    # save model data
    model_evolution.to_csv(f'{datadir}/model_evolution_{paramstr}.csv')
    df_hist_last.to_csv(f'{datadir}/df_hist_last_{paramstr}.csv')
