"""
Phenotype classification from carb-sensitivity BLUPs, validation with phenotypes and prediabetes, and analysis of follow-up HbA1c.

Input : option_a_blup_multi.csv
        df_anthropometrics.csv, df_body.csv, df_hba1c.csv,
        df_prediab.csv, df_a1c.csv
Output: fig5a.png,
        fig5b.png,
        fig5c.png,
        fig5d.png
"""

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

baseline_window_days = 365 
pheno_names = ['low_low', 'low_high', 'high_low', 'high_high']  # magnitude (u0) and sensitivity (u1), low or high
p_cols = ['P_low_low', 'P_low_high', 'P_high_low', 'P_high_high']
formula_p = ' + '.join(p_cols) + ' + age_c + sex - 1'

# read random intercept u0 and slope u1 BLUPs
df_val = pd.read_csv('option_a_blup_multi.csv', float_precision='round_trip').rename(columns={'u0i': 'u0_blup', 'u1i': 'u1_blup'})
blup_array = df_val[['u0_blup', 'u1_blup']].to_numpy()
cov_array = df_val[['cov_00', 'cov_01', 'cov_01', 'cov_11']].to_numpy().reshape(-1, 2, 2)

# posterior probability of each quadrant separeted by u0 = 0 and u1 = 0
np.random.seed(42)
posterior_probs = np.zeros((len(df_val), 4))
for i in range(len(df_val)):
    u0, u1 = np.random.multivariate_normal(blup_array[i], cov_array[i], size=10000).T
    posterior_probs[i] = [((u0 < 0) & (u1 <= 0)).mean(), ((u0 <= 0) & (u1 > 0)).mean(), ((u0 > 0) & (u1 <= 0)).mean(), ((u0 > 0) & (u1 > 0)).mean()]

p_max = posterior_probs.max(axis=1)
assigned = posterior_probs.argmax(axis=1)
confident = p_max >= 0.60
tentative = (p_max >= 0.50) & (p_max < 0.60)
uncertain = p_max < 0.50
df_val[p_cols] = posterior_probs
df_val['assigned'] = assigned
df_val['confident'] = confident

# fig5a: phenotypes in BLUP space
fig, ax = plt.subplots(figsize=(9, 7))
h_unc = ax.scatter('u0_blup', 'u1_blup', data=df_val[uncertain], c='#D0D3D8', alpha=0.45, s=9, edgecolors='none')
h_ten = ax.scatter('u0_blup', 'u1_blup', data=df_val[tentative], c='#E8C24A', alpha=0.75, s=13, edgecolors='black', linewidths=0.25)
h_conf = []
for q, color in enumerate(['#2E7D4F', '#9C5BB0', '#2F7FB8', '#D9534F']):
    h_conf.append(ax.scatter('u0_blup', 'u1_blup', data=df_val[(assigned == q) & confident], c=color, alpha=0.85, s=24, edgecolors='black', linewidths=0.4))
ax.legend([h_unc, h_ten, tuple(h_conf)],
          [f'Uncertain ({uncertain.sum()})', f'Tentative ({tentative.sum()})', f'Confident ({confident.sum()})'],
          handler_map={tuple: HandlerTuple(ndivide=None, pad=1)}, fontsize=9, loc='upper left', bbox_to_anchor=(0, 0.88))
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(0, color='gray', ls='--', lw=0.8)
for q, (text, x, y) in enumerate([('Low magnitude\nLow sensitivity', 0.03, 0.03), ('Low magnitude\nHigh sensitivity', 0.03, 0.97),
                                  ('High magnitude\nLow sensitivity', 0.97, 0.03), ('High magnitude\nHigh sensitivity', 0.97, 0.97)]):
    ax.text(x, y, f'{text}\n(n = {(assigned == q).sum()})', transform=ax.transAxes,
            ha='left' if x < 0.5 else 'right', va='bottom' if y < 0.5 else 'top', fontsize=9, fontweight='bold')
ax.set_xlabel('BLUP intercept (u0)')
ax.set_ylabel('BLUP carb slope (u1)')
ax.set_title('Response Distribution in BLUP Space')
plt.tight_layout()
plt.savefig('fig5a.png', dpi=600, bbox_inches='tight')
plt.show()

# merge person-level phenotype data
for file in ['df_anthropometrics.csv', 'df_hba1c.csv', 'df_body.csv']:
    df_val = df_val.merge(pd.read_csv(file), on='participant_id', how='left')
df_val['age_c'] = df_val['age'] - df_val['age'].mean()

# probability-weighted regression
method_a_models = {marker: smf.ols(f'{marker} ~ {formula_p}', df_val).fit(cov_type='HC3', use_t=True)
                   for marker in ['bt__hba1c_float_value', 'bmi', 'waist_to_hip_ratio', 'body_comp_total_tissue_percent_fat', 'total_scan_vat_mass']}

pw_rows = []
for marker, model in method_a_models.items():
    for a, b in combinations(range(4), 2):
        t_res = model.t_test(f'{p_cols[b]} - {p_cols[a]} = 0')
        pw_rows.append({'marker': marker, 'idx_a': a, 'idx_b': b, 'contrast': f'{pheno_names[b]} vs {pheno_names[a]}', 'diff': t_res.effect.item(), 'p': t_res.pvalue.item()})
pw_df = pd.DataFrame(pw_rows)
pw_df['p_fdr'] = multipletests(pw_df['p'], method='fdr_bh')[1]

# fig5b: adjusted pairwise contrasts for four quadrants
colors = ['#2E7D4F', '#9C5BB0', '#2F7FB8', '#D9534F']
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for ax, (marker, title, ylabel, scale) in zip(axes, [
        ('bt__hba1c_float_value', 'HbA1c', 'HbA1c (%)', 1),
        ('bmi', 'BMI', r'BMI (kg m$^{-2}$)', 1),
        ('waist_to_hip_ratio', 'Waist-to-hip ratio', 'Waist-to-hip ratio', 1),
        ('body_comp_total_tissue_percent_fat', 'Total body fat', 'Total body fat (%)', 100),
        ('total_scan_vat_mass', 'Visceral adipose mass', 'Visceral adipose mass (g)', 1)]):
    means = method_a_models[marker].params.to_numpy()[:4] * scale
    ci = 1.96 * method_a_models[marker].bse.to_numpy()[:4] * scale
    for i in range(4):
        ax.errorbar(i, means[i], yerr=ci[i], fmt='o', color=colors[i], markersize=8, markeredgecolor='white', elinewidth=1.8, capsize=5, capthick=1.8)
    low, top = (means - ci).min(), (means + ci).max()
    y_range = top - low
    # FDR-significant pairs, stacked upwards in reverse order so narrower brackets sit lower
    significant = pw_df[(pw_df['marker'] == marker) & (pw_df['p_fdr'] < 0.05)]
    for k, row in enumerate(significant[::-1].itertuples()):
        y = top + 0.08 * (k + 1) * y_range
        h = 0.02 * y_range
        ax.plot([row.idx_a, row.idx_a, row.idx_b, row.idx_b], [y, y + h, y + h, y], color='black', lw=1)
        ax.text((row.idx_a + row.idx_b) / 2, y + h, '*' * sum(row.p_fdr < cut for cut in (0.05, 0.01, 0.001)), ha='center')
    ax.set_ylim(low - 0.15 * y_range, top + 0.55 * y_range)
    ax.set_xticks([])
    ax.set_title(f'{title}, n = {int(method_a_models[marker].nobs):,}', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
fig.legend([Line2D([], [], marker='o', ls='', color=c, markersize=10) for c in colors],
           ['Low magnitude Low sensitivity', 'Low magnitude High sensitivity', 'High magnitude Low sensitivity', 'High magnitude High sensitivity'],
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False)
plt.tight_layout()
plt.savefig('fig5b.png', dpi=300, bbox_inches='tight')
plt.show()


# read prediabetes info
df_prediab = pd.read_csv('df_prediab.csv')
df_val['is_prediabetic'] = df_val['participant_id'].isin(df_prediab['participant_id']).astype(int)
df_pre = df_val[df_val['is_prediabetic'] == 1]
df_non = df_val[df_val['is_prediabetic'] == 0]

prediab_counts = pd.crosstab(df_val['assigned'], df_val['is_prediabetic']).set_axis(pheno_names)
prediab_pct = 100 * prediab_counts / prediab_counts.sum()

# logistic regression on profile probabilities
model_logit = smf.logit(f'is_prediabetic ~ {formula_p}', df_val).fit(disp=0, maxiter=200)
or_rows = []
for k in range(1, 4):
    t_res = model_logit.t_test(f'{p_cols[k]} - {p_cols[0]} = 0')
    or_rows.append({'profile': pheno_names[k], 'OR': np.exp(t_res.effect.item()), 'OR_low': np.exp(t_res.conf_int()[0, 0]),
                    'OR_high': np.exp(t_res.conf_int()[0, 1]), 'p': t_res.pvalue.item()})
prediab_or = pd.DataFrame(or_rows)

# fig5c: prediabetes in BLUP space with confidence
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(df_non.loc[df_non['confident'], 'u0_blup'], df_non.loc[df_non['confident'], 'u1_blup'],
           alpha=0.10, s=6, c='grey', label=f'Non-pre-diabetic (n={len(df_non)})')
ax.scatter(df_non.loc[~df_non['confident'], 'u0_blup'], df_non.loc[~df_non['confident'], 'u1_blup'], alpha=0.03, s=6, c='grey')
ax.scatter(df_pre.loc[df_pre['confident'], 'u0_blup'], df_pre.loc[df_pre['confident'], 'u1_blup'],
           alpha=0.90, s=30, c='red', edgecolors='darkred', linewidths=0.5, label=f'Pre-diabetic (n={len(df_pre)})', zorder=5)
ax.scatter(df_pre.loc[~df_pre['confident'], 'u0_blup'], df_pre.loc[~df_pre['confident'], 'u1_blup'],
           alpha=0.25, s=30, c='red', edgecolors='darkred', linewidths=0.5, zorder=4)
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.axvline(0, color='black', lw=0.5, ls='--')
ax.set_xlabel('u0 (intercept)')
ax.set_ylabel('u1 (carb slope)')
ax.set_title('BLUP Space')
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('fig5c.png', dpi=300, bbox_inches='tight')
plt.show()


# follow-up HbA1c analysis
df_a1c = pd.read_csv('df_a1c.csv', parse_dates=['a1c_date', 't0'])
baseline_rows = []
for participant_id, draws in df_a1c.groupby('participant_id'):
    pre = draws[(draws['gap'] < 0) & (draws['gap'] >= -baseline_window_days)]
    post = draws[(draws['gap'] > 0) & (draws['gap'] <= baseline_window_days)]
    if not pre.empty:
        row = pre.loc[pre['gap'].idxmax()]
    elif not post.empty and (draws['gap'] > 0).sum() >= 2:
        row = post.loc[post['gap'].idxmin()]
    else:
        continue
    baseline_rows.append({'participant_id': participant_id, 'base_a1c': row['a1c'], 'base_date': row['a1c_date']})

df_long = df_a1c.merge(pd.DataFrame(baseline_rows), on='participant_id')
df_long = (df_long[(df_long['gap'] > 0) & (df_long['a1c_date'] > df_long['base_date'])].merge(df_val[['participant_id', 'age', 'sex', 'bmi', 'u0_blup', 'u1_blup']], on='participant_id'))
df_long = df_long.dropna(subset=['a1c', 'base_a1c', 'age', 'sex', 'bmi', 'u0_blup', 'u1_blup'])

df_long = df_long.join(df_long.groupby('participant_id')[['u0_blup', 'u1_blup']].first().apply(zscore, ddof=1).add_suffix('_z'), on='participant_id')

# follow-up HbA1c on both BLUPs, adjusted for baseline HbA1c, age, sex and BMI
followup_fit = smf.mixedlm('a1c ~ base_a1c + age + C(sex) + bmi + u0_blup_z + u1_blup_z', df_long, groups=df_long['participant_id']).fit(reml=True)
followup_table = pd.DataFrame({'coef': followup_fit.params, 'ci_low': followup_fit.conf_int()[0], 'ci_high': followup_fit.conf_int()[1], 'p': followup_fit.pvalues}).loc[['u0_blup_z', 'u1_blup_z']]

# ANCOVA high high vs low low HbA1c change
plevel = (df_long.groupby('participant_id', sort=False).agg(base_a1c=('base_a1c', 'first'), mean_fu=('a1c', 'mean')).reset_index())
plevel = plevel.merge(df_val[['participant_id', 'assigned', 'confident']], on='participant_id', how='left')
plevel['delta_mean'] = plevel['mean_fu'] - plevel['base_a1c']
plevel['profile'] = plevel['assigned'].map(dict(enumerate(pheno_names)))
term = "C(profile, Treatment(reference='low_low'))[T.high_high]"
panels, ancova_rows = [], []
for label, d in [('All eligible assignments', plevel), ('Confident profiles only', plevel[plevel['confident']])]:
    fit = smf.ols("delta_mean ~ C(profile, Treatment(reference='low_low')) + base_a1c", d).fit()
    delta_adj = d['delta_mean'] - fit.params['base_a1c'] * (d['base_a1c'] - d['base_a1c'].mean())
    ref, cmp = delta_adj[d['profile'] == 'low_low'], delta_adj[d['profile'] == 'high_high']
    panels.append((label, '*' * sum(fit.pvalues[term] < cut for cut in (0.05, 0.01, 0.001)) or 'ns', ref, cmp))
    ancova_rows.append({'analysis_set': label, 'n_low_low': len(ref), 'n_high_high': len(cmp), 'ancova_coef': fit.params[term], 'ci_low': fit.conf_int().loc[term, 0], 'ci_high': fit.conf_int().loc[term, 1], 'p': fit.pvalues[term]})
ancova_df = pd.DataFrame(ancova_rows)

# fig5d: high high vs low low adjusted HbA1c change
fig, ax = plt.subplots(figsize=(9, 5.45))
fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.15)
data = [v for _, _, ref, cmp in panels for v in (ref, cmp)]
bp = ax.boxplot(data, positions=[1, 2, 4, 5], widths=0.7, patch_artist=True, showfliers=False, showmeans=True, meanline=True, medianprops=dict(linewidth=0), meanprops=dict(color='black', linewidth=1.4, linestyle='-'))
for box, color in zip(bp['boxes'], ['#39A432', '#E2071E'] * 2):
    box.set_facecolor(color)

lows = [cap.get_ydata()[0] for cap in bp['caps'][::2]]
tops = [cap.get_ydata()[0] for cap in bp['caps'][1::2]]
span = max(tops) - min(lows)
for x, (label, stars, ref, cmp), top in zip([1, 4], panels, [max(tops[:2]), max(tops[2:])]):
    y = top + 0.08 * span
    ax.plot([x, x, x + 1, x + 1], [y, y + 0.03 * span, y + 0.03 * span, y], color='black', linewidth=1)
    ax.text(x + 0.5, y + 0.04 * span, stars, ha='center', va='bottom')
    ax.text(x + 0.5, -0.12, label, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=11)
ax.set_xticks([1, 2, 4, 5], [f'n = {len(v)}' for v in data], fontsize=11)
ax.set_ylim(min(lows) - 0.1 * span, max(tops) + 0.3 * span)
ax.set_ylabel('Adjusted change in HbA1c')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(handles=[Patch(facecolor='#39A432', edgecolor='black', label='Low-magnitude/ low-responsiveness'),
                   Patch(facecolor='#E2071E', edgecolor='black', label='High-magnitude/ high-responsiveness'),
                   Line2D([], [], color='black', linewidth=1.4, label='Mean')],
          loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, columnspacing=2.0, handletextpad=0.8, frameon=False)
plt.savefig('fig5d.png', dpi=300)
plt.show()
