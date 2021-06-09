from pylab import plt

import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model, metrics

sns.set_palette('colorblind')
sns.set_context('notebook', font_scale=1)

# generate data
n = 1000
alpha_prot = 2 # influence of alpha_prot on x2
maxval = 2
decision_function = 'const'  # 'const' | 'adaptive'
# we have two skill features x1 and x2, and a protected feature x_prot
# the protected feature is correlated with x2
# we draw them from a truncated normal distribution if maxval is not None
x_prot = np.random.choice([0, 1], size=n, replace=True)
if maxval is not None:
    x1 = stats.truncnorm.rvs(-maxval,maxval,size=n)
    x2 = 1/2 * (alpha_prot * (x_prot-0.5) + stats.truncnorm.rvs(-maxval,maxval,size=n))
else:
    x1 = stats.norm.rvs(-maxval, maxval, size=n)
    x2 = 1/2 * (alpha_prot * (x_prot-0.5) + stats.truncnorm.rvs(-maxval,maxval,size=n))
# the real skill is simply the mean of x1 and x2
s_real = (x1 + x2)/2
df = pd.DataFrame({'x1': x1, 'x2': x2, 'x_prot': x_prot, 's_real': s_real})

# # model with protected attribute
# model_with_xpr = linear_model.LinearRegression().fit(df[['x1', 'x_prot']], df['s_real'])
# df['s_pred_with_xpr'] = model_with_xpr.predict(df[['x1', 'x_prot']])
# # model without protected attribute
# model_base = linear_model.LinearRegression().fit(df[['x1']], df['s_real'])
# df['s_pred_base'] = model_base.predict(df[['x1']])


# logistic regression
def train_model(X, y):
    return linear_model.LogisticRegression().fit(X, y)


def train_and_predict(X, y):
    # if there is only once class left in the labels, we make a constant prediction
    # which equals to leaving the labels all the same, and we return 0 for the coefficients
    if len(np.unique(y)) == 1:
        return y, np.array([[0, 0]])
    else:
        model = linear_model.LogisticRegression().fit(X, y)
        return model.predict(X), model.coef_


def real_decision_function(s_real):
    """compute real classes based on real skill"""
    if decision_function == 'const':
        return (s_real > 0).astype(int)
    elif decision_function == 'adaptive':
        return (s_real > s_real.mean()).astype(int)
    else:
        raise ValueError(f'decision function {decision_function} not known')


# define intervention models
# matrix with intervention parameters for the four cases (real class can be 1 or 2, and real class can be 1 or two)
# [[real class 1 & pred class 1, real class 1 & pred class 2],
#  [real class 2 & pred class 1, real class 2 & pred class 2]]
k_matrix = np.array([[1 / 2 / 10, 1 / 10],
                     [1 / 10, 2 / 10]])


def intervention_model(x1, x2, real_class, pred_class, k_matrix):
    """
        update features based on intervention models. works with scalars and vectors
    """
    # we have four parameters, for four cases (real class can be 1 or 2, and real class can be 1 or two)

    k_rcl1_pcl1 = k_matrix[0, 0]
    k_rcl1_pcl2 = k_matrix[0, 1]
    k_rcl2_pcl1 = k_matrix[1, 0]
    k_rcl2_pcl2 = k_matrix[1, 1]
    x1_max = 2
    x2_max = 2
    # make an array with k values for the correct real_class - pred_class combination, then we can use
    # vector operations
    kvec = np.zeros_like(x1)
    kvec[(real_class == 0) & (pred_class == 0)] = k_rcl1_pcl1
    kvec[(real_class == 0) & (pred_class == 1)] = k_rcl1_pcl2
    kvec[(real_class == 1) & (pred_class == 0)] = k_rcl2_pcl1
    kvec[(real_class == 1) & (pred_class == 1)] = k_rcl2_pcl2

    # if x1 or x2 are already above the maximum value that the intervention model can attain,
    # then we do not change them (otherwise they would be reduced when using the same formula)
    # in vector formulation we can achieve this with the maximum function
    x1_new = np.maximum(x1 + (x1_max - x1) * kvec, x1)
    x2_new = np.maximum(x2 + (x2_max - x2) * kvec, x2)

    return x1_new, x2_new


# compute real and predicted classes
df['class'] = real_decision_function(df['s_real'])
model = train_model(df[['x1', 'x_prot']], df['class'])
df['class_pred'] = model.predict(df[['x1', 'x_prot']])

conf_matr = metrics.confusion_matrix(df['class'], df['class_pred'])
# compute cases that are incorrectly classified, and would be correctly classified if their prot_attr
# would be different
wrong_cases = df[df['class'] != df['class_pred']].copy()
n_wrong = len(wrong_cases)
assert (n_wrong == conf_matr[0, 1] + conf_matr[1, 0])
print(f'{n_wrong} are incorrectly classified')
# flip labels
wrong_cases['x_prot_flipped'] = wrong_cases['x_prot'].replace({1: 0, 0: 1})
preds_alt = model.predict(wrong_cases[['x1', 'x_prot_flipped']])
n_wrong_flipped = sum(preds_alt != wrong_cases['class'])
n_flipped_better = wrong_cases - n_wrong_flipped
print(f'when flipping the protected attribute of the wrong cases, only {n_wrong_flipped} remain wrong')
print(f'{n_flipped_better} of the wrong cases would thus get a different (=correct) classification if the'
      f'protected attribute is changed ')

df_init = df.copy()


def step_model(df):
    df['x1'], df['x2'] = intervention_model(df['x1'], df['x2'], df['class'], df['class_pred'], k_matrix)
    # compute new real classes
    df['s_real'] = (df['x1'] + df['x2'])/2
    df['class'] = real_decision_function(df['s_real'])
    # pred_model = train_model(df[['x1', 'x_prot']], df['class'])
    df['class_pred'], coefs = train_and_predict(df[['x1', 'x_prot']], df['class'])

    return df.copy(), coefs


n_steps = 50
res = []
res_summary = []
for i in range(n_steps):
    df, coefs = step_model(df)
    summary = pd.DataFrame({  # 'step':i,
        'x1_mean': df['x1'].mean(),
        'x2_mean': df['x2'].mean(),
        's_mean': df['s_real'].mean(),
        'x1_group1': df['x1'][df['x_prot'] == 0].mean(),
        'x2_group1': df['x2'][df['x_prot'] == 0].mean(),
        'x1_group2': df['x1'][df['x_prot'] == 1].mean(),
        'x2_group2': df['x2'][df['x_prot'] == 1].mean(),
        's_group1': df['s_real'][df['x_prot'] == 0].mean(),
        's_group2': df['s_real'][df['x_prot'] == 1].mean(),
        'coef2': coefs[0, 1],
    }, index=[i])
    res.append(df)
    res_summary.append(summary)

res_summary = pd.concat(res_summary)
df_final = df.copy()

sns.jointplot('x1', 'x2', data=df_init, hue='x_prot')
sns.catplot(x='x_prot', hue='class', data=df_init, kind='count')
plt.tight_layout()

sns.jointplot('x1', 'x2', data=df_final, hue='x_prot')
sns.catplot(x='x_prot', hue='class', data=df_final, kind='count')
plt.tight_layout()

plt.figure()
ax = plt.subplot(221)
res_summary[['x1_mean', 'x2_mean']].plot(alpha=0.8, ax=ax)
ax = plt.subplot(222)
res_summary[['x1_group1', 'x2_group1', 'x1_group2', 'x2_group2']].plot(alpha=0.8, ax=ax)
ax = plt.subplot(223)
res_summary[['s_mean', 's_group1', 's_group2']].plot(alpha=0.8, ax=ax)
ax = plt.subplot(224)
res_summary[['coef2']].plot(alpha=0.8, ax=ax)
plt.suptitle(f'decision_function: {decision_function}, k={k_matrix}')
