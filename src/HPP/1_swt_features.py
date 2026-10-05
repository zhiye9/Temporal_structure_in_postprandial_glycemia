"""
SWT feature extraction and traditional metrics from 3-hour postprandial CGM curves.

Input: df_postprandial_cgm.csv.
Each rows represent a single meal, with 13 timepoints of the postprandial CGM curve, mael time (hour of a day) and meal nutritent composition. 
Columns: participant_id,participant_id, meal_id, pp_cgm_glucose_min_{0,15,...,180}, carbohydrate_g, dietary_fiber_g, protein_g, lipid_g, alcohol_g, hour_of_day


Output: df_swt.csv. Same meal-level data with added columns swt_db2_energy_{cA2,cD2,cD1}, peak_rise, iauc, peak_time. 
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

# The df_postprandial_cgm.csv is generated using simglucose (https://github.com/jxx123/simglucose)
df_swt = pd.read_csv('Synthetic_data/df_postprandial_cgm.csv')
grid_minutes = np.arange(0, 181, 15)

# SWT (db2, level 2)
cgm_matrix = df_swt[[f"pp_cgm_glucose_min_{t}" for t in grid_minutes]].values.astype(float)
cgm_matrix = cgm_matrix - cgm_matrix[:, [0]]
cgm_padded = np.pad(cgm_matrix, pad_width=((0, 0), (1, 2)), mode="symmetric")

coeffs = {'cA2': np.zeros_like(cgm_matrix), 'cD2': np.zeros_like(cgm_matrix), 'cD1': np.zeros_like(cgm_matrix)}
for i in range(len(cgm_padded)):
    result = pywt.swt(cgm_padded[i], "db2", level=2)

    coeffs['cA2'][i] = result[0][0][1:14]    
    coeffs['cD2'][i] = result[0][1][1:14]    
    coeffs['cD1'][i] = result[1][1][1:14]    

for band, c in coeffs.items():
    df_swt[f"swt_db2_energy_{band}"] = np.sum(c**2, axis=1)

# traditional metrics
df_swt['peak_rise'] = cgm_matrix.max(axis=1)
df_swt['iauc'] = np.trapz(np.maximum(cgm_matrix, 0), grid_minutes, axis=1)
df_swt['peak_time'] = cgm_matrix.argmax(axis=1) * 15

# Fig2a:boxplot of feauture distributions
labels = ['peak Rise', 'iAUC', 'peak Time', 'cA2', 'cD2', 'cD1']
colors = ['#A9C6DD', '#6BA0BF', '#38617A', '#5B2C6F', '#d62728', '#ff7f0e']

data = [
    np.log(df_swt['peak_rise'].dropna().values),
    np.log(df_swt['iauc'].dropna().values),
    np.log(df_swt['peak_time'].dropna().values),   
    np.log(df_swt['swt_db2_energy_cA2'].dropna().values),
    np.log(df_swt['swt_db2_energy_cD2'].dropna().values),
    np.log(df_swt['swt_db2_energy_cD1'].dropna().values),
]

fig, ax = plt.subplots(figsize=(9, 7))

bp = ax.boxplot(
    data,
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    medianprops=dict(color='#333333', linewidth=1.6),
    whiskerprops=dict(color='#666666', linewidth=1.2),
    capprops=dict(color='#666666', linewidth=1.2),
    flierprops=dict(marker='o', markersize=3, markerfacecolor='none',
                    markeredgecolor='#999999', alpha=0.5),
)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('#555555')
    patch.set_linewidth(1.2)
    patch.set_alpha(0.9)

ax.set_ylabel("ln(value)")
ax.tick_params(axis='y', left=True, labelleft=True)
ax.spines['left'].set_visible(True)

fig.tight_layout()
fig.savefig('Fig2a.png', dpi=300, bbox_inches='tight')
plt.show()

# compute available carbohydrate, winsorize and log transform
df_swt['available_carbohydrate_g'] = (
    df_swt['carbohydrate_g'] - df_swt['dietary_fiber_g'])

df_swt['has_alcohol'] = (df_swt['alcohol_g'] > 0).astype(int)
df_swt['peak_rise'] = df_swt['peak_rise'].clip(-50, 250)
df_swt['iauc'] = df_swt['iauc'].clip(-3000, 20000)
for resp in ['swt_db2_energy_cA2', 'swt_db2_energy_cD2', 'swt_db2_energy_cD1']:
    df_swt[resp] = df_swt[resp].clip(df_swt[resp].quantile(0.005), df_swt[resp].quantile(0.995))

for resp in ['peak_rise', 'iauc', 'swt_db2_energy_cA2', 'swt_db2_energy_cD2', 'swt_db2_energy_cD1']:
    df_swt[f"{resp}_log"] = np.log(df_swt[resp])

# meal timing category and carbohydrate centering within cluster (CWC)
hour = df_swt['hour_of_day']
df_swt['_meal_category'] = np.select(
    [(hour >= 5) & (hour < 11), (hour >= 11) & (hour < 17), (hour >= 17) & (hour < 23), (hour >= 23) | (hour < 5)],
    ['breakfast', 'lunch', 'dinner', 'late_night'], default='unknown')

person_carb_mean = df_swt.groupby('participant_id')['available_carbohydrate_g'].transform('mean')
person_means = df_swt.groupby('participant_id')['available_carbohydrate_g'].mean()
df_swt['carb_between'] = (person_carb_mean - person_means.mean()) / person_means.std()
df_swt['carb_within'] = df_swt['available_carbohydrate_g'] - person_carb_mean
df_swt['carb_within'] = df_swt['carb_within'] / df_swt['carb_within'].std()

# save the preprocessed data for the next step
df_swt.to_csv('df_swt.csv', index=False)
