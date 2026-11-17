"""
Cross-level interactions analysis for independent and joint modeling of person-level moderators.

Input : df_swt.csv
        df_anthropometrics.csv, df_body.csv, df_hba1c.csv, microbiome_pcoa.csv,
        GCTA/greml_outputs_v2/all_cvblups_zscore_v2.csv,
        section3b_blup_reliability_summary.csv
Output: fig4a.png,
        fig4b.png,
        fig4c.png,
        figS5.png
"""

import warnings

import numpy as np
import pandas as pd
import patsy
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from utils import fit_mixedlm, fit_joint_model

# merge person-level moderators
df_model = (pd.read_csv('df_swt.csv')
            .merge(pd.read_csv('df_anthropometrics.csv'), on='participant_id', how='left')
            .merge(pd.read_csv('df_body.csv'), on='participant_id', how='left')
            .merge(pd.read_csv('df_hba1c.csv'), on='participant_id', how='left')
            .merge(pd.read_csv('GCTA/greml_outputs_v2/all_cvblups_zscore_v2.csv'), on='participant_id', how='left')
            .merge(pd.read_csv('microbiome_pcoa.csv'), on='participant_id', how='left'))


# normalize moderators
df_model['sex'] = df_model['sex'].astype(float)
z_cols = {'age': 'age_z', 'bmi': 'bmi_z', 'waist_to_hip_ratio': 'waist_to_hip_ratio_z',
          'body_comp_total_tissue_percent_fat': 'body_comp_total_tissue_percent_fat_z',
          'total_scan_vat_mass': 'total_scan_vat_mass_z', 'bt__hba1c_float_value': 'bt__hba1c_float_value_z',
          'pco1': 'pcoa1_z', 'pco2': 'pcoa2_z',
          'gblup_cA2': 'gblup_cA2_z', 'gblup_cD2': 'gblup_cD2_z', 'gblup_cD1': 'gblup_cD1_z',
          'gblup_iAUC': 'gblup_iAUC_z', 'gblup_peak_rise': 'gblup_peak_rise_z', 'gblup_peak_time': 'gblup_peak_time_z'}
df_model = df_model.join(df_model.groupby('participant_id')[list(z_cols)].first()
                         .apply(zscore, ddof=1, nan_policy='omit').rename(columns=z_cols), on='participant_id')

stage3_baseline = {}
for _, row in pd.read_csv('section3b_blup_reliability_summary.csv', float_precision='round_trip').iterrows():
    stage3_baseline[row['response']] = {
        'resp_col': row['response'], 'var_u0': row['var_u0'], 'var_u1': row['var_u1'], 'var_e': row['var_e'],
        'fe_params': {col[4:]: row[col] for col in row.index if col.startswith('fe__')},
    }

# fit moderator
formula_nutrients = "carb_within + carb_between + protein_g + lipid_g + dietary_fiber_g + has_alcohol + C(_meal_category)"
base_cols = ['participant_id', 'carb_within', 'carb_between', 'protein_g', 'lipid_g', 'dietary_fiber_g', 'has_alcohol']

response_to_gblup = {
    'swt_db2_energy_cA2_log': 'gblup_cA2',
    'swt_db2_energy_cD2_log': 'gblup_cD2',
    'swt_db2_energy_cD1_log': 'gblup_cD1',
    'iauc_log':               'gblup_iAUC',
    'peak_rise_log':          'gblup_peak_rise',
    'peak_time':              'gblup_peak_time',
}

individual_tests = [
    ('Demographics', ['age_z', 'sex']),
    ('BMI', ['bmi_z']),
    ('WaistHipRatio', ['waist_to_hip_ratio_z']),
    ('BodyCompFat', ['body_comp_total_tissue_percent_fat_z']),
    ('VAT', ['total_scan_vat_mass_z']),
    ('HbA1c', ['bt__hba1c_float_value_z']),
    ('Microbiome', ['pcoa1_z', 'pcoa2_z']),
]

# phase A: independent screening
phase_a_results = {}

for resp_name, baseline in stage3_baseline.items():
    resp_col = baseline['resp_col']
    phase_a_results[resp_name] = {}

    for test_name, test_vars in individual_tests + [('gBLUP', [response_to_gblup[resp_name] + '_z'])]:
        formula = (f"{resp_col} ~ {formula_nutrients} + " + " + ".join(test_vars) + " + " + " + ".join(f"carb_within:{v}" for v in test_vars))

        df_fit = df_model.dropna(subset=[resp_col] + base_cols + test_vars)
        meal_counts = df_fit.groupby('participant_id').size()
        df_fit = df_fit[df_fit['participant_id'].isin(meal_counts.index)].copy()

        fit, status = fit_mixedlm(formula, df_fit, df_fit['participant_id'], "~carb_within")

        G = fit.cov_re
        var_u0 = float(G.iloc[0, 0])
        var_u1 = float(G.iloc[1, 1]) if G.shape[0] >= 2 else 0
        cov_u01 = float(G.iloc[0, 1]) if G.shape[0] >= 2 else 0
        cor_u01 = cov_u01/np.sqrt(var_u0 * var_u1) if var_u0 > 0 and var_u1 > 0 else 0

        interaction_results = {}
        for moderator in test_vars:
            term = f"carb_within:{moderator}"
            interaction_results[term] = {
                'coef': fit.fe_params[term], 'se': fit.bse[term],
                'z': fit.tvalues[term], 'p': fit.pvalues[term],
            }

        phase_a_results[resp_name][test_name] = {
            'status': status,
            'var_u0': var_u0, 'var_u1': var_u1, 'cov_u01': cov_u01, 'cor_u01': cor_u01, 'var_e': fit.scale,
            'n_obs': len(df_fit), 'n_groups': df_fit['participant_id'].nunique(),
            'pct_tau2_slope_from_baseline': (100 * (var_u1 - baseline['var_u1'])/baseline['var_u1']
                                             if baseline['var_u1'] > 0 else 0),
            'formula': formula,
            'fe_params': fit.fe_params.to_dict(), 'fe_pvalues': fit.pvalues.to_dict(),
            'interaction_results': interaction_results,
            'main_vars': test_vars, 'interact_vars': test_vars,
        }

# phase B: joint modeling
phase_b_tests = []
for test_name in [name for name, _ in individual_tests] + ['gBLUP']:
    significant = False
    for response_results in phase_a_results.values():
        result = response_results[test_name]
        if result['status'].startswith('OK'):
            p_values = [info['p'] for info in result['interaction_results'].values()]
            if any(p < 0.05 for p in p_values):
                significant = True
    if significant:
        phase_b_tests.append(test_name)

phase_b_interact = {}
for resp_name in stage3_baseline:
    test_variables = dict(individual_tests)
    test_variables['gBLUP'] = [response_to_gblup[resp_name] + '_z']
    phase_b_interact[resp_name] = []
    for test_name in phase_b_tests:
        phase_b_interact[resp_name].extend(test_variables[test_name])

phase_b_results = {}
for resp_name, baseline in stage3_baseline.items():
    phase_b_results[resp_name] = fit_joint_model(
        df_model, baseline, phase_b_interact[resp_name],
        formula_nutrients, base_cols)

# R2 decomposition
rows = []
for resp_name, s3 in stage3_baseline.items():
    specs = ([('baseline_stage3', f"{s3['resp_col']} ~ {formula_nutrients}", s3)]
             + [(test_name, res['formula'], res) for test_name, res in phase_a_results[resp_name].items()
                if res['status'].startswith('OK')])
    for test_name, formula_str, res in specs:
        _, X = patsy.dmatrices(formula_str, df_model, return_type='dataframe', NA_action='drop')
        var_fixed = float(np.var(X.values @ np.array([res['fe_params'][c] for c in X.columns]), ddof=0))
        total = var_fixed + res['var_u0'] + res['var_u1'] + res['var_e']
        rows.append({'r2_marginal': var_fixed/total,
                     'r2_conditional': (var_fixed + res['var_u0'] + res['var_u1']) / total,
                     'gap': (res['var_u0'] + res['var_u1']) / total,
                     'response': resp_name, 'test': test_name})
df_r2 = pd.DataFrame(rows)

# fig4a: phase A heatmap 
resp_order = ['swt_db2_energy_cA2_log', 'swt_db2_energy_cD2_log', 'swt_db2_energy_cD1_log',
              'peak_rise_log', 'iauc_log', 'peak_time']

coef_T = np.array([[info['coef'] for res in phase_a_results[resp].values() for info in res['interaction_results'].values()]
                   for resp in resp_order])
pval_T = np.array([[info['p'] for res in phase_a_results[resp].values() for info in res['interaction_results'].values()]
                   for resp in resp_order])

fig, ax = plt.subplots(figsize=(10.4, 4.3))
im = ax.imshow(np.clip(np.sign(coef_T) * -np.log10(pval_T), -4, 4), cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
for i, j in zip(*np.where(pval_T < 0.05)):
    ax.text(j, i, '*' * sum(pval_T[i, j] < cut for cut in (0.05, 0.01, 0.001)), ha='center', va='center', fontsize=14, fontweight='bold')
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.5, zorder=3))
ax.set_xticks(range(10), ['Age', 'Sex', 'BMI', 'Waist-Hip\nRatio', 'Body\nFat %', 'VAT\nMass', 'HbA1c', 'PCo1', 'PCo2', 'gBLUP'])
ax.set_yticks(range(6), ['cA2 Energy', 'cD2 Energy', 'cD1 Energy', 'Peak Rise', 'iAUC', 'Peak Time'])
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
ax.axhline(2.5, color='white', linewidth=2.5)
for xpos in [1.5, 3.5, 5.5, 6.5, 8.5]:
    ax.axvline(xpos, color='white', linewidth=1.5)

top = ax.secondary_xaxis('top')
top.set_xticks([0.5, 2.5, 4.5, 6, 7.5, 9], ['Demographics', 'Anthropometrics', 'Body Composition', 'Glycemic', 'Microbiome', 'Genetic'])
top.tick_params(length=0)
top.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, ticks=[-4, -2, 0, 2, 4])
plt.tight_layout()
plt.savefig('fig4a.png', dpi=300)
plt.show()


# fig4b: phase B coefficient attenuation
mod_label = {'age_z': 'Age', 'sex': 'Sex', 'bmi_z': 'BMI', 'waist_to_hip_ratio_z': 'Waist-Hip Ratio',
             'body_comp_total_tissue_percent_fat_z': 'Body Fat %', 'total_scan_vat_mass_z': 'VAT Mass',
             'bt__hba1c_float_value_z': 'HbA1c', 'pcoa1_z': 'Microbiome PCo1', 'pcoa2_z': 'Microbiome PCo2'}

rows = []
for resp, label in zip(['swt_db2_energy_cA2_log', 'swt_db2_energy_cD2_log', 'swt_db2_energy_cD1_log',
                        'peak_rise_log', 'iauc_log', 'peak_time'],
                       ['cA2 Energy', 'cD2 Energy', 'cD1 Energy', 'Peak Rise', 'iAUC', 'Peak Time']):
    sd_y = df_model[resp].std()
    for res in phase_a_results[resp].values():
        for term, info in res['interaction_results'].items():
            if info['p'] < 0.05:
                rows.append({'resp_name': resp, 'resp': label, 'mod': mod_label.get(term.replace('carb_within:', ''), 'gBLUP'),
                             'coef_a': info['coef'] / sd_y, 'p_a': info['p'],
                             'coef_b': phase_b_results[resp]['fe_params'][term] / sd_y,
                             'p_b': phase_b_results[resp]['fe_pvalues'][term]})
df_rows = pd.DataFrame(rows)

mod_color = {'Age': '#9CA3AF', 'Sex': '#B0B5BC', 'BMI': '#2F7FB8', 'Waist-Hip Ratio': '#6BA0BF', 'Body Fat %': '#E39A62',
             'VAT Mass': '#C77A3E', 'HbA1c': '#D9534F', 'Microbiome PCo1': '#2E7D4F', 'Microbiome PCo2': '#6FB08A', 'gBLUP': '#CA983D'}

panels = [df_rows[df_rows['resp_name'].str.startswith('swt')], df_rows[~df_rows['resp_name'].str.startswith('swt')]]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 1.5 + 0.4 * max(len(df) + df['resp'].nunique() for df in panels)), sharex=True)
for ax, df_panel in zip(axes, panels):
    ticks, headers = [], []
    for resp, df_group in df_panel.groupby('resp', sort=False):
        headers.append(len(ticks))
        ticks.append(resp)
        for _, row in df_group.iterrows():
            y = -len(ticks)
            sig = row['p_b'] < 0.05
            ticks.append(row['mod'])
            ax.plot([row['coef_a'], row['coef_b']], [y, y], color=mod_color[row['mod']], linewidth=1.5, alpha=0.8 if sig else 0.3)
            ax.scatter(row['coef_a'], y, color=mod_color[row['mod']], marker='o', s=80, zorder=3)
            ax.scatter(row['coef_b'], y, color=mod_color[row['mod']], marker='D', s=70, alpha=1 if sig else 0.4, zorder=3)
            ax.text(1.02, y, 'Retained' if sig else 'Not retained', transform=ax.get_yaxis_transform(), va='center',
                    fontstyle='italic', fontweight='bold' if sig else 'normal')
    ax.set_yticks(-np.arange(len(ticks)), ticks)
    for i in headers:
        ax.get_yticklabels()[i].set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.axvline(0, color='#9CA3AF', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Standardized interaction coefficient')
fig.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='#4B5563', markersize=9, label='Independent screen'),
                    Line2D([0], [0], marker='D', color='w', markerfacecolor='#4B5563', markersize=8, label='Joint modeling'),
                    Line2D([0], [0], marker='D', color='w', markerfacecolor='#4B5563', markersize=8, alpha=0.3, label='n.s.')],
           loc='upper right', ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
plt.savefig('fig4b.png', dpi=300)
plt.show()


# fig4c: R2 decomposition for wavelet features
fig, axes = plt.subplots(1, 3, figsize=(10.4, 4), sharey=True)
axes[0].set_ylim(0, df_r2.loc[df_r2['response'].str.startswith('swt_db2_energy'), 'r2_conditional'].max() * 100 * 1.15)
for ax, resp, label in zip(axes, ['swt_db2_energy_cA2_log', 'swt_db2_energy_cD2_log', 'swt_db2_energy_cD1_log'],
                           ['cA2 Energy', 'cD2 Energy', 'cD1 Energy']):
    df_resp = df_r2[df_r2['response'] == resp].set_index('test')[['r2_marginal', 'gap']] * 100
    best = df_resp.drop('baseline_stage3').sum(axis=1).idxmax()
    df_pair = df_resp.loc[['baseline_stage3', best]]
    ax.bar([0, 1], df_pair['r2_marginal'], 0.62, color='#6AAB5E', label='$R^2_{\\mathrm{marginal}}$')
    ax.bar([0, 1], df_pair['gap'], 0.62, bottom=df_pair['r2_marginal'], color='#3B6E8F', label='Participant-level variance')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='center', color='white', fontsize=9)
    ax.bar_label(ax.containers[1], labels=[f'{v:.1f}%' for v in df_pair.sum(axis=1)], fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1], ['Base', '+ ' + {'WaistHipRatio': 'Waist-Hip', 'BodyCompFat': 'Body Fat %', 'VAT': 'VAT Mass'}.get(best, best)])
    ax.set_xlabel(label)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('% of Total Variance Explained')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', ncol=2, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig('fig4c.png', dpi=300)
plt.show()

# figS5: R2 decomposition for wavelet features and traditional metrics
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
axes[0, 0].set_ylim(0, df_r2['r2_conditional'].max() * 100 * 1.15)
for ax, resp, label in zip(axes.flat,
                           ['swt_db2_energy_cA2_log', 'swt_db2_energy_cD2_log', 'swt_db2_energy_cD1_log',
                            'peak_rise_log', 'iauc_log', 'peak_time'],
                           ['cA2 Energy', 'cD2 Energy', 'cD1 Energy', 'Peak Rise', 'iAUC', 'Peak Time']):
    # baseline first, then one bar per moderator, in % of total variance
    df_resp = df_r2[df_r2['response'] == resp].set_index('test').loc[
        ['baseline_stage3', 'Demographics', 'BMI', 'WaistHipRatio', 'BodyCompFat', 'VAT', 'HbA1c', 'Microbiome', 'gBLUP'],
        ['r2_marginal', 'gap']] * 100
    ax.bar(range(9), df_resp['r2_marginal'], 0.62, color='#6AAB5E', label='$R^2_{\\mathrm{marginal}}$')
    ax.bar(range(9), df_resp['gap'], 0.62, bottom=df_resp['r2_marginal'], color='#3B6E8F', label='Participant-level variance')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='center', color='white', fontsize=9)
    ax.bar_label(ax.containers[1], labels=[f'{v:.1f}%' for v in df_resp.sum(axis=1)], fontsize=10, fontweight='bold')
    ax.set_xticks(range(9))
    ax.set_xticklabels(['Base', '+ Demographics', '+ BMI', '+ Waist-Hip', '+ Body Fat %', '+ VAT Mass', '+ HbA1c',
                        '+ Microbiome', '+ gBLUP'], rotation=40, ha='right')
    ax.set_xlabel(label)
    ax.spines[['top', 'right']].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel('% of Total Variance Explained')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig('figS5.png', dpi=300, bbox_inches='tight')
plt.show()