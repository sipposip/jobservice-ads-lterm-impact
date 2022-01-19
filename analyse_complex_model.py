import os
from pylab import plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.ioff()

plotdir = './plots_complexmodel/allscenarios/'
plotdir2 = './plots_complexmodel/singlescenarios/'
datadir = './data_complexmodel/'
for d in (plotdir, plotdir2):
    if not os.path.isdir(d):
        os.mkdir(d)


def savefig(figname):
    plt.savefig(f'{figname}.svg')
    plt.savefig(f'{figname}.pdf')
    plt.savefig(f'{figname}.png', dpi=400)


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
labormarket_bias = 0  # run with 0 and 2

if labormarket_bias == 0:
    labormarket_bias_string = 'unbiased'
elif labormarket_bias > 0:
    labormarket_bias_string = 'biased'
else:
    labormarket_bias_string = 'biased towards underprivileged'

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
                      jobmarket_function_loc, jobmarket_function_scale, scale_factor, labormarket_bias)])

# lineplots, colored scenarios, one plot for each modeltype
figsize = (7, 3)
colors = sns.color_palette('colorblind', n_colors=7)
for metric in ('BGSD', 'coef1', 'coef2', 'coef2_standardized'):
    for modeltype in modeltypes:
        plt.figure(figsize=figsize)
        sns.lineplot('time', metric, hue='scenario', data=model_evolution_all.query('modeltype==@modeltype'))
        plt.title(f'modeltype={modeltype}, labormarket={labormarket_bias_string}')
        sns.despine()
        if metric == 'BGSD':
            plt.ylim(0, 0.45)
        savefig(f'{plotdir}/{metric}_vs_time_allscens_{paramstr}_{modeltype}_complexmodel')

# barplots
xvals = np.unique(model_evolution_end_agg['scenario'])
_x = np.arange(len(xvals))
# this approach is only valid if the order of the scenarios is the same for both model types (which
# is the case here if nothing is changed above, but we check it anyway)

assert (np.array_equal(model_evolution_end_agg['scenario'][model_evolution_end_agg['modeltype'] == 'base'],
                       model_evolution_end_agg['scenario'][model_evolution_end_agg['modeltype'] == 'full']))
width = 0.4
for metric in ('BGSD', 'BGaccuracyD', 'BGprecisionD', 'BGrecallD', 's_all', 'coef2_standardized'):
    # at end of simulation
    plt.figure(figsize=figsize)
    plt.bar(_x - width / 2,
            model_evolution_end_agg[metric][model_evolution_end_agg['modeltype'] == 'base'],
            width=width, color=colors, hatch='//', label='base', edgecolor='grey',
            linewidth=5)
    plt.bar(_x + width / 2,
            model_evolution_end_agg[metric][model_evolution_end_agg['modeltype'] == 'full'],
            width=width, color=colors, hatch='--', label='full', edgecolor='grey',
            linewidth=5)
    plt.xticks(_x, xvals)
    leg = plt.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(0)
    plt.ylabel(metric)
    sns.despine()
    plt.title('end of simulation')
    savefig(f'{plotdir}/barplot_{metric}_{paramstr}_complexmodel')

metric = 'fraction_of_lowpros_correctly_highpros_after_flipping'
plt.figure(figsize=figsize)
plt.bar(_x - width / 2,
        model_evolution_end_agg[metric][model_evolution_end_agg['modeltype'] == 'base'],
        width=width, color=colors, hatch='//', label='base', edgecolor='grey',
        linewidth=5)
plt.bar(_x + width / 2,
        model_evolution_end_agg[metric][model_evolution_end_agg['modeltype'] == 'full'],
        width=width, color=colors, hatch='--', label='full', edgecolor='grey',
        linewidth=5)
plt.xticks(_x, xvals)
plt.ylabel('counterfactual fraction')
sns.despine()
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(0)
plt.ylim(0, 0.34)
plt.title('end of simulation')
savefig(f'{plotdir}/barplot_{metric}_{paramstr}_complexmodel')

for metric in ('BGSD', 's_all'):
    # difference between start and end
    plt.figure(figsize=figsize)
    plt.bar(_x - width / 2,
            model_evolution_diff[metric][model_evolution_diff['modeltype'] == 'base'],
            width=width, color=colors, hatch='//', label='base', edgecolor='grey',
            linewidth=5)
    plt.bar(_x + width / 2,
            model_evolution_diff[metric][model_evolution_diff['modeltype'] == 'full'],
            width=width, color=colors, hatch='--', label='full', edgecolor='grey',
            linewidth=5)
    plt.xticks(_x, xvals)
    leg = plt.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(0)
    plt.ylabel(metric)
    sns.despine()
    if metric == 'BGSD':
        plt.ylim(-0.25, 0)
    plt.title('difference end - start of simulation')
    savefig(f'{plotdir}/barplot_diff_{metric}_{paramstr}_complexmodel')

# multipanel plots, single plot for each scenario modeltype configuration
for scenario in np.unique(model_evolution_all['scenario']):
    for modeltype in modeltypes:
        model_evolution = model_evolution_all.query('(scenario==@scenario) & (modeltype==@modeltype)')
        n_rows = 4
        n_cols = 2
        fig = plt.figure(figsize=(13, 7))

        ax1 = plt.subplot(n_rows, n_cols, 1)
        model_evolution[['s_priv', 's_upriv', 's_all']].plot(ax=ax1)
        sns.despine()
        plt.ylabel('$s_{real}$')

        ax = plt.subplot(n_rows, n_cols, 2)
        model_evolution['BGSD'].plot(ax=ax)
        plt.ylabel('BGSD')
        sns.despine()

        ax = plt.subplot(n_rows, n_cols, 3)
        model_evolution['mean_Tu_priv_current'].plot(ax=ax, label='priv')
        model_evolution['mean_Tu_upriv_current'].plot(ax=ax, label='upriv')
        plt.legend()
        plt.ylabel('$T_{u}$')
        sns.despine()

        ax = plt.subplot(n_rows, n_cols, 4)
        model_evolution['BGTuD_current'].plot(ax=ax)
        plt.ylabel('BGTuD_current')
        sns.despine()

        ax = plt.subplot(n_rows, n_cols, 5)
        model_evolution[['accuracy', 'recall', 'precision']].plot(ax=ax)
        # since accuracy, recall and precision start with NANs, the plotting
        # function omits them and starts the xaxis only at the first valid value
        # therfore we have to set the xlims
        plt.xlim(*ax1.get_xlim())
        sns.despine()

        ax = plt.subplot(n_rows, n_cols, 6)
        model_evolution['frac_upriv'].plot(ax=ax)
        sns.despine()
        plt.ylabel('frac_upriv')

        ax = plt.subplot(n_rows, n_cols, 7)
        # model_evolution['inbal_waiting'].plot(ax=ax)
        # plt.ylabel('inbal_waiting')
        plt.plot(model_evolution.eval('n_waiting_priv / n_active_priv'), label='priv')
        plt.plot(model_evolution.eval('n_waiting_upriv / n_active_upriv'), label='upriv')
        plt.legend()
        plt.ylabel('fraction in waiting')
        sns.despine()
        plt.xlabel('t')

        ax = plt.subplot(n_rows, n_cols, 8)
        model_evolution['fraction_of_lowpros_correctly_highpros_after_flipping'].plot(ax=ax)
        sns.despine()
        plt.ylabel('counterfactual fraction')
        plt.xlim(*ax1.get_xlim())
        plt.xlabel('t')
        plt.suptitle(f'scenario {scenario} modeltype {modeltype} , labormarket {labormarket_bias_string}')
        plt.tight_layout(w_pad=0, h_pad=0)
        savefig(f'{plotdir2}/onescenario_multipanel{paramstr}_{scenario}_{modeltype}')
