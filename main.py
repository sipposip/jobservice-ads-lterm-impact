from pylab import plt

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
import scipy.special

n = 1000
alpha_prot = 2
x1 = np.random.normal(0, 1, size=n)
x_prot = np.random.choice([0, 1], size=n, replace=True)
x2 = alpha_prot * (x_prot - 0.5) + np.random.normal(0, 1, size=n)

s_real = x1 + x2

df = pd.DataFrame({'x1': x1, 'x2': x2, 'x_prot': x_prot, 's_real': s_real})

# model with protected attribute
model_with_xpr = linear_model.LinearRegression().fit(df[['x1', 'x_prot']], df['s_real'])
df['s_pred_with_xpr'] = model_with_xpr.predict(df[['x1', 'x_prot']])
# model without protected attribute
model_base = linear_model.LinearRegression().fit(df[['x1']], df['s_real'])
df['s_pred_base'] = model_base.predict(df[['x1']])

# sns.lmplot('x1', 'x2', data=df)
# sns.lmplot('x1', 'x2', data=df, hue='x_prot')
# sns.lmplot('x1', 's_real', data=df, hue='x_prot')
# sns.lmplot('x2', 's_real', data=df, hue='x_prot')
# sns.lmplot('x1', 's_pred_with_xpr', data=df, hue='x_prot')
# sns.lmplot('s_pred_with_xpr', 's_real', data=df, hue='x_prot')
# plt.title(f"R2={np.corrcoef(df['s_pred_with_xpr'], df['s_real'])[0, 1]:.2f}")
# plt.tight_layout()
# sns.lmplot('s_pred_base', 's_real', data=df, hue='x_prot')
# plt.title(f"R2={np.corrcoef(df['s_pred_base'], df['s_real'])[0, 1]:.2f}")
# plt.tight_layout()
#
# # now define classes: class0: s<0, class1: s>0
df['class'] = (df['s_real'] > 0).astype(int)


# plt.figure()
# sns.histplot(x='x_prot', hue='class', data=df, multiple="dodge")


# logistic regression
def train_model(X, y):
    return linear_model.LogisticRegression().fit(X, y)


logmodel_with_xpr = train_model(df[['x1', 'x_prot']], df['class'])
df['class_pred'] = logmodel_with_xpr.predict(df[['x1', 'x_prot']])

# model without protected attribute
logmodel_base = linear_model.LogisticRegression().fit(df[['x1']], df['class'])
df['class_pred_base'] = logmodel_base.predict(df[['x1']])

conf_matr = metrics.confusion_matrix(df['class'], df['class_pred'])
# compute cases that are incorrectly classified, and would be correctly classified if their prot_attr
# would be different
wrong_cases = df[df['class'] != df['class_pred']].copy()
n_wrong = len(wrong_cases)
assert (n_wrong == conf_matr[0, 1] + conf_matr[1, 0])
print(f'{n_wrong} are incorrectly classified')
# flip labels
wrong_cases['x_prot_flipped'] = wrong_cases['x_prot'].replace({1: 0, 0: 1})
preds_alt = logmodel_with_xpr.predict(wrong_cases[['x1', 'x_prot_flipped']])
n_wrong_flipped = sum(preds_alt != wrong_cases['class'])
n_flipped_better = wrong_cases - n_wrong_flipped
print(f'when flipping the protected attribute of the wrong cases, only {n_wrong_flipped} remain wrong')
print(f'{n_flipped_better} of the wrong cases would thus get a different (=correct) classification if the'
      f'protected attribute is changed ')


# define intervention models

def intervention_model(x1, x2, real_class, pred_class):
    """
        update features based on intervention models. works with scalars and vectors
    """
    # class agnostic model
    # we have four parameters, for four cases (real class can be 1 or 2, and real class can be 1 or two)
    k_rcl1_pcl1 = 2 / 10
    k_rcl1_pcl2 = 1 / 10
    k_rcl2_pcl1 = 1 / 10
    k_rcl2_pcl2 = 4 / 10
    x1_max = 2
    x2_max = 2
    # make an array with k values for the correct real_class - pred_class combination, then we can use
    # vector operations
    kvec = np.zeros_like(x1)
    kvec[(real_class == 0) & (pred_class == 0)] = k_rcl1_pcl1
    kvec[(real_class == 0) & (pred_class == 1)] = k_rcl1_pcl2
    kvec[(real_class == 1) & (pred_class == 0)] = k_rcl2_pcl1
    kvec[(real_class == 1) & (pred_class == 1)] = k_rcl2_pcl2

    x1_new = x1 + (x1_max - x1) * kvec
    x2_new = x2 + (x2_max - x2) * kvec

    return x1_new, x2_new


df_init = df.copy()

df = df_init.copy()


def step_model(df):
    df['x1'], df['x2'] = intervention_model(df['x1'], df['x2'], df['class'], df['class_pred'])
    # compute new real classes
    df['s_real'] = df['x1'] + df['x2']
    df['class'] = (df['s_real'] > df['s_real'].mean()).astype(int)
    pred_model = train_model(df[['x1', 'x_prot']], df['class'])
    df['class_pred'] = pred_model.predict(df[['x1', 'x_prot']])

    return df.copy(), pred_model


n_steps = 50
res = []
res_summary = []
for i in range(n_steps):
    df, model = step_model(df)
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
        'coef2': model.coef_[0, 1],
    }, index=[i])
    res.append(df)
    res_summary.append(summary)

res_summary = pd.concat(res_summary)

plt.figure()
ax = plt.subplot(221)
res_summary[['x1_mean', 'x2_mean']].plot(alpha=0.8, ax=ax)
ax = plt.subplot(222)
res_summary[['x1_group1', 'x2_group1', 'x1_group2', 'x2_group2']].plot(alpha=0.8, ax=ax)
ax = plt.subplot(223)
res_summary[['s_mean', 's_group1', 's_group2']].plot(alpha=0.8, ax=ax)
ax = plt.subplot(224)
res_summary[['coef2']].plot(alpha=0.8, ax=ax)
