"""Progressive mixed-effects models from null model to nutrients and meal timing, add random slope to carb, then extend to fat, protein and fiber.

Input: df_swt.csv from 1_swt_features.py. Meal-level cgm features and nutrient data.

Output: section3b_blup_reliability_summary.csv for 3_cross_level_interaction.py
        option_a_blup_multi.csv for 4_phenotyping_validation.py. 
        Fig2a.png.
        Figs4.png.
"""
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

from utils import fit_mixedlm, cross_nutrient_mixedlm, lrt_test

# read data
df_model = pd.read_csv('df_swt.csv')

# mixed LM formulas
formula_base = "carb_within + carb_between + protein_g + lipid_g + dietary_fiber_g + has_alcohol"
formula_nutrients = formula_base + " + C(_meal_category)"

meal_counts = df_model.groupby('participant_id').size().rename('n_meals')

# progressive modeling
progressive_results = {}  
stage3_baseline = {}  

response_vars = [
    'peak_rise_log',
    'iauc_log',
    'peak_time',
    'swt_db2_energy_cA2_log',
    'swt_db2_energy_cD2_log',
    'swt_db2_energy_cD1_log',
]

for response in response_vars:
    df_resp = df_model

    # Stage 1: null model
    fit_s1, _ = fit_mixedlm(f"{response} ~ 1", df_resp, df_resp['participant_id'], "~1")
    var_u0_s1, var_e_s1 = float(fit_s1.cov_re.iloc[0, 0]), fit_s1.scale

    # Stage 2a: nutrient fixed effects
    fit_s2a, _ = fit_mixedlm(f"{response} ~ {formula_base}", df_resp, df_resp['participant_id'], "~1")
    var_u0_s2a, var_e_s2a = float(fit_s2a.cov_re.iloc[0, 0]), fit_s2a.scale

    # Stage 2b: + meal timing
    fit_s2b, _ = fit_mixedlm(f"{response} ~ {formula_nutrients}", df_resp, df_resp['participant_id'], "~1")
    var_u0_s2b, var_e_s2b = float(fit_s2b.cov_re.iloc[0, 0]), fit_s2b.scale

    # Stage 3: + random carb slope
    fit_s3, _ = fit_mixedlm(f"{response} ~ {formula_nutrients}", df_resp, df_resp['participant_id'], "~carb_within")
    G = fit_s3.cov_re
    var_u0_s3, var_e_s3 = float(G.iloc[0, 0]), fit_s3.scale
    var_u1_s3 = float(G.iloc[1, 1]) if G.shape[0] >= 2 else 0
    cov_u01_s3 = float(G.iloc[0, 1]) if G.shape[0] >= 2 else 0
    cor_u01_s3 = cov_u01_s3 / np.sqrt(var_u0_s3 * var_u1_s3) if var_u0_s3 > 0 and var_u1_s3 > 0 else 0

    # LRT test boundary-corrected
    lrt_stat = max(2*(fit_s3.llf - fit_s2b.llf), 0.0)
    p_mix = 0.5 * (1 - stats.chi2.cdf(lrt_stat, df=1)) + 0.5 * (1 - stats.chi2.cdf(lrt_stat, df=2))

    progressive_results[response] = {
        'Stage1_null': {
            'var_u0': var_u0_s1, 'var_e': var_e_s1, 'icc': var_u0_s1 / (var_u0_s1 + var_e_s1),
            'n_obs': len(df_resp), 'n_groups': df_resp['participant_id'].nunique(),
        },
        'Stage2a_nutrients': {
            'var_u0': var_u0_s2a, 'var_e': var_e_s2a, 'icc': var_u0_s2a / (var_u0_s2a + var_e_s2a),
            'delta_var_e_pct': 100 * (1 - var_e_s2a / var_e_s1),
            'delta_var_u0_pct': 100 * (1 - var_u0_s2a / var_u0_s1),
            'fe_params': fit_s2a.fe_params.to_dict(), 'fe_pvalues': fit_s2a.pvalues.to_dict()},
        'Stage2b_timing': {
            'var_u0': var_u0_s2b, 'var_e': var_e_s2b, 'icc': var_u0_s2b / (var_u0_s2b + var_e_s2b),
            'delta_var_e_pct_vs_2a': 100 * (1 - var_e_s2b / var_e_s2a),
            'delta_var_e_pct_vs_null': 100 * (1 - var_e_s2b / var_e_s1),
            'fe_params': fit_s2b.fe_params.to_dict(), 'fe_pvalues': fit_s2b.pvalues.to_dict()},
        'Stage3_random_slope': {
            'var_u0': var_u0_s3, 'var_u1': var_u1_s3, 'cor_u01': cor_u01_s3, 'var_e': var_e_s3,
            'icc': var_u0_s3 / (var_u0_s3 + var_e_s3),
            'lrt_stat': lrt_stat, 'lrt_p_mixture': p_mix},
    }
    stage3_baseline[response] = {
        'fit': fit_s3, 'resp_col': response, 'is_log': response.endswith('_log'),
        'var_u0': var_u0_s3, 'var_u1': var_u1_s3, 'cov_u01': cov_u01_s3, 'cor_u01': cor_u01_s3, 'var_e': var_e_s3,
        'llf': fit_s3.llf,
        'n_obs': len(df_resp), 'n_groups': df_resp['participant_id'].nunique(),
        'gamma10': fit_s3.fe_params.get('carb_within', 0.0),
        'lrt_stat': lrt_stat, 'lrt_p_chi2': 1 - stats.chi2.cdf(lrt_stat, df=2), 'lrt_p_mixture': p_mix,
        'fe_params': fit_s3.fe_params.to_dict(), 'fe_pvalues': fit_s3.pvalues.to_dict(),
    }

# extract BLUPs and reliability
blup_dfs = {}
baseline_rows = []
for response, info in stage3_baseline.items():
    blup_rows = []
    for pid, re_vals in info['fit'].random_effects.items():
        blup_rows.append({
            'participant_id': pid,
            'u0i': float(re_vals.iloc[0]),
            'u1i': float(re_vals.iloc[1]),
        })
    df_blup = pd.DataFrame(blup_rows)
    df_blup = df_blup.merge(meal_counts.reset_index(), on='participant_id', how='left')
    df_blup['reliability_slope'] = (info['var_u1'] / (info['var_u1'] + info['var_e'] / df_blup['n_meals'])
                                    if info['var_u1'] > 0 else 0.0)
    df_blup['reliability_intercept'] = (info['var_u0'] / (info['var_u0'] + info['var_e'] / df_blup['n_meals'])
                                        if info['var_u0'] > 0 else 0.0)
    df_blup['total_slope'] = info['gamma10'] + df_blup['u1i']
    blup_dfs[response] = df_blup

    baseline_rows.append({
        'response': response,
        'var_u0': info['var_u0'], 'var_u1': info['var_u1'], 'var_e': info['var_e'],
        **{f'fe__{term}': coefficient for term, coefficient in info['fe_params'].items()},
    })
# save BLUPs results
stage3_baseline_table = pd.DataFrame(baseline_rows)
stage3_baseline_table.to_csv('section3b_blup_reliability_summary.csv', index=False)


# Figure 1b: variance structure
resp_order = ['swt_db2_energy_cA2_log', 'swt_db2_energy_cD2_log', 'swt_db2_energy_cD1_log',
                'peak_rise_log', 'iauc_log', 'peak_time']

# plot fig2b
icc_pct = np.array([100 * progressive_results[r]['Stage1_null']['icc'] for r in resp_order])
nutrients_pct = np.array([100 * max(progressive_results[r]['Stage1_null']['var_e'] - progressive_results[r]['Stage2b_timing']['var_e'], 0)
                            / progressive_results[r]['Stage1_null']['var_e'] for r in resp_order])
slope_pct = np.array([100 * progressive_results[r]['Stage3_random_slope']['var_u1']
                        / sum(progressive_results[r]['Stage3_random_slope'][k] for k in ['var_u0', 'var_u1', 'var_e'])
                        for r in resp_order])
variance_structure_values = pd.DataFrame({
    'response': resp_order, 'icc_pct': icc_pct,
    'nutrients_timing_pct': nutrients_pct, 'carb_slope_pct': slope_pct,
})

fig, ax = plt.subplots(figsize=(10, 5.5))
y = np.arange(len(resp_order))
ax.barh(y + 0.25, icc_pct, 0.25, color='#3B6E8F', edgecolor='#0F172A', linewidth=0.3, label='Between-person (ICC)')
ax.barh(y, nutrients_pct, 0.25, color='#D96B3B', edgecolor='#0F172A', linewidth=0.6, label='Nutrients + meal timing')
ax.barh(y - 0.25, slope_pct, 0.25, color='#9b3f5c', edgecolor='#0F172A', linewidth=0.6, label='Carbs sensitivity')
ax.axhline(y=2.5, color='#9CA3AF', linewidth=0.8, linestyle='--')   

ax.set_yticks(y)
ax.set_yticklabels(['cA2 Energy', 'cD2 Energy', 'cD1 Energy', 'Peak Rise', 'iAUC', 'Peak Time'], fontsize=17.5)
ax.set_xlabel('% of Variance', fontsize=17.5)

ax.invert_yaxis()

ax.legend(fontsize=15, loc='lower right', bbox_to_anchor=(0.98, 0.02))
plt.tight_layout()
plt.savefig('fig2b.png', dpi=300, bbox_inches='tight')
plt.show()


# plot fig s4, BLUP reliability curve
median_meals = meal_counts.median().astype(float)
n_max = max(100, int(np.ceil(median_meals)) + 10)
n_range = np.arange(1, n_max + 1)
fig, ax = plt.subplots(figsize=(8, 6))
for response, label, style, color, lw in zip(
        resp_order,
        ['cA2 (low-freq)', 'cD2 (mid-freq)', 'cD1 (high-freq)', 'Peak Rise', 'iAUC', 'Peak Time'],
        ['-', '-', '-', '--', '--', '--'],                     
        ['#b7282e', '#d16d5b', '#eabaa1', '#313772', '#326db6', '#75b5dc'],
        [2.5, 1.8, 1.8, 2.0, 1.5, 1.5]):
    ax.plot(n_range, stage3_baseline[response]['var_u1'] / (stage3_baseline[response]['var_u1'] + stage3_baseline[response]['var_e'] / n_range),
            style, color=color, linewidth=lw, label=label)
for r_target in [0.5, 0.7]:
    ax.axhline(y=r_target, color='#CCCCCC', linewidth=0.8, linestyle=':')
    ax.text(n_max + 2, r_target, f'R = {r_target}', va='center', ha='left', fontsize=8, color='#999999')
ax.axvline(x=median_meals, color='#CCCCCC', linewidth=0.8, linestyle=':')
ax.text(median_meals + 1.5, 0.05, f'Current\nmedian\n({median_meals:g} meals)',
        fontsize=8, color='#999999', va='bottom')
ax.set_xlabel('Number of Meals per Person', fontsize=11)
ax.set_ylabel('BLUP Reliability of Carb Sensitivity Estimate', fontsize=11)
ax.set_title('Estimation Reliability Increases with Meal Count', fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(1, n_max)
ax.set_ylim(0, 0.85)
ax.legend(fontsize=8.5, loc='lower right', framealpha=0.95, edgecolor='#CCCCCC', ncol=2)
plt.tight_layout()
plt.savefig('Figs4.png', dpi=300, bbox_inches='tight')
plt.show()

# cross-nutrient analysis
df_cross = df_model.copy()
nutrient_cols = {'carb': 'available_carbohydrate_g', 'protein': 'protein_g', 'fat': 'lipid_g', 'fiber': 'dietary_fiber_g'}
pairwise_pairs = [('carb', 'fat'), ('carb', 'protein'), ('fat', 'protein')]

meal_counts = df_cross.groupby('participant_id').size().rename('n_meals')
n_persons = df_cross['participant_id'].nunique()

# CWC decomposition for all nutrients
cwc_rows = []
for nut, col in nutrient_cols.items():
    person_means = df_cross.groupby('participant_id')[col].mean()
    pm_map = df_cross['participant_id'].map(person_means)
    person_variances = df_cross.groupby('participant_id')[col].var(ddof=1)
    pooled_within_var = person_variances.mean()
    df_cross[f"{nut}_within"] = (df_cross[col] - pm_map) / np.sqrt(pooled_within_var)
    df_cross[f"{nut}_between"] = (pm_map - person_means.mean()) / person_means.std()
    cwc_rows.append({
        'nutrient': nut, 'raw_col': col,
        'grand_mean': df_cross[col].mean(), 'grand_sd': df_cross[col].std(),
        'person_mean_min': person_means.min(), 'person_mean_max': person_means.max(),
        'person_mean_sd': person_means.std(), 'pooled_within_sd': np.sqrt(pooled_within_var),
        'pct_between': 100*person_means.var()/(person_means.var() + pooled_within_var),
        'pct_within': 100*pooled_within_var/(person_means.var() + pooled_within_var),
        'n_persons_within_sd_lt_5g': int((df_cross.groupby('participant_id')[col].std() < 5).sum()),
        'n_persons': n_persons,
    })

# cross-nutrient formula
formula_fe = ("carb_within + carb_between + protein_within + protein_between + "
              "fat_within + fat_between + fiber_within + fiber_between + has_alcohol + C(_meal_category)")
ri_results = {}
ri_rows = []
for response in response_vars:
    df_fit = df_cross
    fit, _ = cross_nutrient_mixedlm(f"{response} ~ {formula_fe}", df_fit, groups='participant_id', re_formula="~1")
    var_u0, var_e = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    ri_results[response] = {
        'fit': fit, 'df_fit': df_fit,
        'var_u0': var_u0, 'var_e': var_e,
        'icc': var_u0/(var_u0 + var_e) if var_u0 + var_e > 0 else np.nan,
        'llf': fit.llf, 'n_obs': len(df_fit), 'n_grp': df_fit['participant_id'].nunique(),
    }
    ri_rows.append({'response': response, **{k: v for k, v in ri_results[response].items() if k not in ('fit', 'df_fit')}})
ri_reference = pd.DataFrame(ri_rows)

# cross-nutrient option A: single-nutrient random slopes
option_a_results = {nut: {} for nut in nutrient_cols} 
for nut in nutrient_cols:
    slope_var = f"{nut}_within"
    for response in response_vars:
        ref = ri_results[response]

        t0 = time.time()
        fit, status = cross_nutrient_mixedlm(f"{response} ~ {formula_fe}", ref['df_fit'].copy(),
                                  groups='participant_id', re_formula=f"~{slope_var}")
        elapsed = time.time() - t0
        print(elapsed)

        if fit is None or status.startswith('FAILED') or status == 'INTERCEPT_ONLY':
            option_a_results[nut][response] = {'elapsed': elapsed}
            continue

        # G matrix: correlation between intercept and slope variances
        re_cov = fit.cov_re
        idx = list(re_cov.index).index(slope_var)
        var_u0, var_u1 = float(re_cov.iloc[0, 0]), float(re_cov.iloc[idx, idx])
        sd0, sd1 = np.sqrt(max(var_u0, 0)), np.sqrt(max(var_u1, 0))
        gamma = fit.fe_params.get(slope_var, np.nan)
        lrt_stat, lrt_p_chi2, lrt_p_mix = lrt_test(fit.llf, ref['llf'], n_extra=2)

        option_a_results[nut][response] = {
            'fit': fit,
            'var_u0': var_u0, 'var_u1': var_u1,
            'cor_u01': float(re_cov.iloc[0, idx])/(sd0 * sd1) if sd0 > 0 and sd1 > 0 else np.nan,
            'var_e': float(fit.scale), 'gamma': gamma, 'slope_sd': sd1,
            'range_lo': gamma - 1.96*sd1, 'range_hi': gamma + 1.96*sd1,  
            'lrt_stat': lrt_stat, 'lrt_p_chi2': lrt_p_chi2, 'lrt_p_mix': lrt_p_mix,
            'llf': fit.llf,
            'n_obs': ref['n_obs'], 'n_grp': ref['n_grp'], 'elapsed': elapsed,
        }


a_rows = []
for response in response_vars:
    for nut in nutrient_cols:
        r = option_a_results[nut][response]
        row = {'response': response, 'nutrient': nut}
        if 'var_u1' in r:
            row.update({k: r[k] for k in ['var_u0', 'var_u1', 'slope_sd', 'cor_u01', 'var_e', 'gamma', 'range_lo', 'range_hi',
                                          'lrt_stat', 'lrt_p_chi2', 'lrt_p_mix', 'n_obs', 'n_grp']})
            row['sig'] = 'ns'
            if r['lrt_p_mix'] < 0.001:
                row['sig'] = '***'
            elif r['lrt_p_mix'] < 0.01:
                row['sig'] = '**'
            elif r['lrt_p_mix'] < 0.05:
                row['sig'] = '*'
            row.update({f'meals_needed_R{tgt}': (r['var_e'] / r['var_u1']) * (tgt / (1 - tgt))
                        if r['var_u1'] > 0 else np.nan for tgt in [0.5, 0.7, 0.8]})
        a_rows.append(row)
option_a_summary = pd.DataFrame(a_rows)

# BLUP and reliability
blup_multi = {} 
for nut in nutrient_cols:
    for response in response_vars:
        r = option_a_results[nut][response]
        if r.get('fit') is None:
            continue
        conditional_cov = r['fit'].random_effects_cov
        blup_rows = []
        for pid, effects in r['fit'].random_effects.items():
            blup_rows.append({
                'participant_id': pid,
                'u0i': float(effects.iloc[0]),
                f'u1i_{nut}': float(effects.iloc[1]),
                f'total_slope_{nut}': r['gamma'] + float(effects.iloc[1]),
                'n_meals': int(meal_counts[pid]),
                f'reliability_{nut}': r['var_u1'] / (r['var_u1'] + r['var_e'] / int(meal_counts[pid])),
                'reliability_intercept': r['var_u0'] / (r['var_u0'] + r['var_e'] / int(meal_counts[pid])),
                'cov_00': float(conditional_cov[pid].iloc[0, 0]),
                'cov_01': float(conditional_cov[pid].iloc[0, 1]),
                'cov_11': float(conditional_cov[pid].iloc[1, 1]),
            })
        blup_df = pd.DataFrame(blup_rows)
        blup_multi.setdefault(nut, {})[response] = blup_df

blup_columns = ['participant_id', 'u0i', 'u1i', 'n_meals', 'nutrient', 'response',
                'cov_00', 'cov_01', 'cov_11']
classification_blups = blup_multi.get('carb', {}).get('swt_db2_energy_cA2_log')
blup_export = (classification_blups.rename(columns={'u1i_carb': 'u1i'})
               .assign(nutrient='carb', response='swt_db2_energy_cA2_log')[blup_columns]
               if classification_blups is not None else pd.DataFrame(columns=blup_columns))
blup_export.to_csv('option_a_blup_multi.csv', index=False)

# cross-nutrient option B: pairwise random slopes
option_b_results = {}   
for nut_a, nut_b in pairwise_pairs:
    sv_a, sv_b = f"{nut_a}_within", f"{nut_b}_within"
    pair_key = f"{nut_a}_{nut_b}"
    option_b_results[pair_key] = {}
    for response in response_vars:
        ref = ri_results[response]

        t0 = time.time()
        fit, status = cross_nutrient_mixedlm(f"{response} ~ {formula_fe}", ref['df_fit'].copy(),
                                  groups='participant_id', re_formula=f"~{sv_a} + {sv_b}")
        elapsed = time.time() - t0

        if fit is None or status.startswith('FAILED') or status == 'INTERCEPT_ONLY':
            option_b_results[pair_key][response] = {}
            continue

        re_cov = fit.cov_re
        labels = list(re_cov.index)
        idx_a, idx_b = labels.index(sv_a), labels.index(sv_b)
        var_u1_a, var_u1_b = float(re_cov.iloc[idx_a, idx_a]), float(re_cov.iloc[idx_b, idx_b])
        sd_a, sd_b = np.sqrt(max(var_u1_a, 0)), np.sqrt(max(var_u1_b, 0))

        lrt_stat, lrt_p_chi2, lrt_p_mix = lrt_test(fit.llf, ref['llf'], n_extra=5)
        lrt_vs_a = dict(zip(['stat', 'p_std', 'p_mix'], lrt_test(fit.llf, option_a_results[nut_a][response]['llf'], 2))) if 'llf' in option_a_results[nut_a][response] else {}
        lrt_vs_b = dict(zip(['stat', 'p_std', 'p_mix'], lrt_test(fit.llf, option_a_results[nut_b][response]['llf'], 2))) if 'llf' in option_a_results[nut_b][response] else {}

        option_b_results[pair_key][response] = {
            'fit': fit,
            'var_u0': float(re_cov.iloc[0, 0]), 'var_u1_a': var_u1_a, 'var_u1_b': var_u1_b,
            'cor_slopes': float(re_cov.iloc[idx_a, idx_b]) / (sd_a * sd_b) if sd_a > 0 and sd_b > 0 else np.nan,
            'cov_slopes': float(re_cov.iloc[idx_a, idx_b]), 'var_e': float(fit.scale),
            'lrt_stat': lrt_stat, 'lrt_p_mix': lrt_p_mix,
            'lrt_vs_a': lrt_vs_a, 'lrt_vs_b': lrt_vs_b, 'elapsed': elapsed,
        }

# cross-nutrient option C:  all nutrient random slopes
slope_vars_c = ['carb_within', 'fat_within', 'protein_within']
c_rows = []
for response in response_vars:
    ref = ri_results[response]

    t0 = time.time()
    fit, status = cross_nutrient_mixedlm(f"{response} ~ {formula_fe}", ref['df_fit'].copy(),
                              groups='participant_id', re_formula="~" + " + ".join(slope_vars_c))
    elapsed = time.time() - t0

    if fit is None or status.startswith('FAILED') or status == 'INTERCEPT_ONLY':
        c_rows.append({'response': response})
        continue

    re_cov = fit.cov_re
    labels = list(re_cov.index)
    row = {'response': response, 'var_u0': float(re_cov.iloc[0, 0]),
           'var_e': float(fit.scale), 'elapsed': elapsed}
    for sv in slope_vars_c:
        i = labels.index(sv)
        row[f'var_u1_{sv[:-7]}'] = float(re_cov.iloc[i, i])
        row[f'cor_u0_{sv[:-7]}'] = (float(re_cov.iloc[0, i]) / np.sqrt(float(re_cov.iloc[0, 0]) * float(re_cov.iloc[i, i]))
                                   if re_cov.iloc[0, 0] > 0 and re_cov.iloc[i, i] > 0 else np.nan)
    for k, sv_i in enumerate(slope_vars_c):
        for sv_j in slope_vars_c[k + 1:]:
            i, j = labels.index(sv_i), labels.index(sv_j)
            row[f'cor_{sv_i[:-7]}_{sv_j[:-7]}'] = (float(re_cov.iloc[i, j]) / np.sqrt(float(re_cov.iloc[i, i]) * float(re_cov.iloc[j, j]))
                                                  if re_cov.iloc[i, i] > 0 and re_cov.iloc[j, j] > 0 else np.nan)
    c_rows.append(row)

option_c_summary = pd.DataFrame(c_rows)

# nutrient and frequency matrix
frequency_rows = []
for nut in nutrient_cols:
    for response in response_vars:
        result = option_a_results[nut][response]
        p = result.get('lrt_p_mix', np.nan)
        sig = '---'
        if 'lrt_p_mix' in result:
            sig = 'ns'
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
        frequency_rows.append({
            'nutrient': nut, 'response': response,
            'var_u1': result.get('var_u1', np.nan), 'lrt_p_mix': p, 'sig': sig,
        })
nutrient_frequency_matrix = pd.DataFrame(frequency_rows)
