import os
import sys
from pylab import plt
import seaborn as sns
import numpy as np
import pandas as pd

plotdir = './plots_complexmodel/allscenarios/'
datadir = './data_complexmodel/'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)

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
labormarket_bias=2

modeltypes = ('full', 'base')
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

model_evolution_all = []
df_hist_last_all = []
for config in configs:
    scenario = config['scenario']

    k_matrix = config['k_matrix']

    for modeltype in modeltypes:
        # parameterstring string for filenames
        paramstr = '_'.join(
            [str(e) for e in (alpha_prot, tsteps, n_spinup, n_retain_from_spinup, delta_T_u, T_u_max, class_boundary,
                              jobmarket_function_loc, jobmarket_function_scale, scale_factor, modeltype,
                              labormarket_bias, scenario)])

        model_evolution = pd.read_csv(f'{datadir}/model_evolution_{paramstr}.csv', index_col=0)
        df_hist_last = pd.read_csv(f'{datadir}/df_hist_last_{paramstr}.csv', index_col=0)
        model_evolution['scenario'] = scenario
        model_evolution['modeltype'] = modeltype
        model_evolution['time'] = model_evolution.index
        df_hist_last['scneario'] = scenario
        df_hist_last['modeltype'] = modeltype
        model_evolution_all.append(model_evolution)
        df_hist_last_all.append(df_hist_last)

model_evolution_all = pd.concat(model_evolution_all)
df_hist_last_all = pd.concat(df_hist_last_all)

# compute compund metrics
model_evolution_all['coef2_standardized'] = model_evolution_all.eval('coef2/(coef1+coef2)')

# extract last timesteps for averaging.
n_last = 200
# we need to select all timesteps between the highest and highes-n_last timesteps
model_evolution_end = model_evolution_all[model_evolution_all['time'] > np.max(model_evolution_all['time']) - n_last]
# average over time, per scenario
model_evolution_end_agg = model_evolution_end.groupby(['scenario', 'modeltype']).mean()
# the same, for the retained spinup period
model_evolution_start = model_evolution_all[
    model_evolution_all['time'] < np.min(model_evolution_all['time']) + n_retain_from_spinup]
model_evolution_start_agg = model_evolution_start.groupby(['scenario', 'modeltype']).mean()

# compute diff
model_evolution_diff = model_evolution_end_agg - model_evolution_start_agg

# reset multi index (scenario and modeltype) to normal columns, this is handier for plotting
model_evolution_start_agg.reset_index(inplace=True)
model_evolution_end_agg.reset_index(inplace=True)
model_evolution_diff.reset_index(inplace=True)
# paramstr without scenario and modeltype
paramstr = '_'.join(
    [str(e) for e in (alpha_prot, tsteps, n_spinup, n_retain_from_spinup, delta_T_u, T_u_max, class_boundary,
                      jobmarket_function_loc, jobmarket_function_scale, scale_factor,labormarket_bias)])




# lineplots, colored scenarios, one plot for each modeltype
figsize = (7, 3)
colors = sns.color_palette('colorblind', n_colors=7)
for metric in ('BGSD', 'coef1','coef2','coef2_standardized'):
    for modeltype in modeltypes:
        plt.figure(figsize=figsize)
        sns.lineplot('time', metric, hue='scenario', data=model_evolution_all.query('modeltype==@modeltype'))
        plt.title(f'modeltype={modeltype}')
        sns.despine()
        if metric == 'BGSD':
            plt.ylim(0.1, 0.45)
        plt.savefig(f'{plotdir}/{metric}_vs_time_allscens_{paramstr}_{modeltype}_complexmodel.svg')
        plt.savefig(f'{plotdir}/{metric}_vs_time_allscens_{paramstr}_{modeltype}_complexmodel.png')
        plt.savefig(f'{plotdir}/{metric}_vs_time_allscens_{paramstr}_{modeltype}_complexmodel.pdf')

# barplots
xvals = np.unique(model_evolution_end_agg['scenario'])
_x = np.arange(len(xvals))
# this approach is only valid if the order of the scenarios is the same for both model types (which
# is the case here if nothing is changed above, but we check it anyway)

assert (np.array_equal(model_evolution_end_agg['scenario'][model_evolution_end_agg['modeltype'] == 'base'],
                       model_evolution_end_agg['scenario'][model_evolution_end_agg['modeltype'] == 'full']))
width = 0.4
for metric in ('BGSD', 'BGaccuracyD', 'BGprecisionD', 'BGrecallD','s_all', 'coef2_standardized'):

    # at end of simulation
    plt.figure(figsize=figsize)
    plt.bar(_x - width / 2,
            model_evolution_end_agg[metric][model_evolution_end_agg['modeltype'] == 'base'],
            width=width, color=colors, hatch='//', label='base')
    plt.bar(_x + width / 2,
            model_evolution_end_agg[metric][model_evolution_end_agg['modeltype'] == 'full'],
            width=width, color=colors, hatch='--', label='full')
    plt.xticks(_x, xvals)
    plt.legend()
    plt.ylabel(metric)
    sns.despine()
    plt.title('end of simulation')
    plt.savefig(f'{plotdir}/barplot_{metric}_{paramstr}_complexmodel.svg')
    plt.savefig(f'{plotdir}/barplot_{metric}_{paramstr}_complexmodel.pdf')
    # the hatch patterns often lead to problems with svg or pdf, therefore we also save png
    plt.savefig(f'{plotdir}/barplot_{metric}_{paramstr}_complexmodel.png', dpi=400)

for metric in ('BGSD','s_all'):
    # difference between start and end
    plt.figure(figsize=figsize)
    plt.bar(_x - width / 2,
            model_evolution_diff[metric][model_evolution_diff['modeltype'] == 'base'],
            width=width, color=colors, hatch='//', label='base')
    plt.bar(_x + width / 2,
            model_evolution_diff[metric][model_evolution_diff['modeltype'] == 'full'],
            width=width, color=colors, hatch='--', label='full')
    plt.xticks(_x, xvals)
    plt.legend()
    plt.ylabel(metric)
    sns.despine()
    plt.title('difference end - start of simulation')
    plt.savefig(f'{plotdir}/barplot_diff_{metric}_{paramstr}_complexmodel.svg')
    plt.savefig(f'{plotdir}/barplot_diff_{metric}_{paramstr}_complexmodel.pdf')
    plt.savefig(f'{plotdir}/barplot_diff_{metric}_{paramstr}_complexmodel.png', dpi=400)
