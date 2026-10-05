import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import find_peaks
import pywt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# read OGTT df
df_ogtt = pd.read_csv('df_ogtt.csv', index_col=0)

ogtt_times = np.array([0, 15, 30, 60, 90, 120, 150, 240]) 
subject_ids = df_ogtt.index.to_numpy()

glucose_cols = df_ogtt[[f'Glucose_{float(t)}' for t in ogtt_times]].to_numpy(float)    
insulin_cols = df_ogtt[[f'Insulin_{float(t)}' for t in ogtt_times]].to_numpy(float)    
cpep_cols = df_ogtt[[f'C-peptid_{float(t)}' for t in ogtt_times]].to_numpy(float)
para_cols = df_ogtt[['Paracetamol_15.0', 'Paracetamol_30.0']].to_numpy(float) 

# interpolated to 15 mins and cut to 180 mins
glucose_grid = PchipInterpolator(ogtt_times, glucose_cols, axis=1)(np.arange(0, 181, 15))

# t0 correction
glucose_grid = glucose_grid - glucose_grid[:, [0]]      

# swt
padded = np.pad(glucose_grid, ((0, 0), (1, 2)), mode='symmetric')      
ca2, cd2, cd1 = pywt.swt(padded, 'db2', level=2, trim_approx=True, axis=1)   
ca2_energy = np.sum(ca2[:, 1:14]**2, axis=1)                         
cd2_energy = np.sum(cd2[:, 1:14]**2, axis=1)
cd1_energy = np.sum(cd1[:, 1:14]**2, axis=1)

# metabolic indices
# HOMA-IR
g0, g30 = glucose_cols[:, 0], glucose_cols[:, 2]           
i0 = insulin_cols[:, 0]                                    
homa_ir = (i0 / 6.945) * g0 / 22.5

# CGI
delta_g = np.where(g30 - g0 > 0, g30 - g0, np.nan)         # G30 - G0 <= 0 or missing -> NaN
cgi = (cpep_cols[:, 2] - cpep_cols[:, 0])/1000/delta_g

# Matsuda
matsuda_0_240_raw = 10000 / np.sqrt((g0 * 18.0156)*(i0/6.945)*(np.nanmean(glucose_cols, axis=1)*18.0156)*(np.nanmean(insulin_cols, axis=1)/6.945))

#hiri = sqrt(auc_g_0_30 * auc_i_0_30)
hiri_sqrt = np.sqrt((0.5 * (glucose_cols[:, 0] + glucose_cols[:, 2])/2)*(0.5*(insulin_cols[:, 0] + insulin_cols[:, 2])/2))

# misi was computed according to the MISI Calculator
misi_calc = np.full(len(glucose_cols), np.nan)
for s in range(len(glucose_cols)):
    if np.isnan(glucose_cols[s, [0, 2, 3, 4, 5]]).any() or np.isnan(insulin_cols[s, [0, 2, 3, 4, 5]]).any():
        continue

    g_spline = CubicSpline([-30, -15, -7, 0, 30, 60, 90, 120], glucose_cols[s, [0, 0, 0, 0, 2, 3, 4, 5]])
    g = g_spline(np.arange(121))

    i_spline = CubicSpline([-30, -15, -7, 0, 30, 60, 90, 120], insulin_cols[s, [0, 0, 0, 0, 2, 3, 4, 5]])
    i_bar = i_spline(np.arange(121)).mean()

    max_idx = int(np.argmax(g))                         
    g_max = g[max_idx]
    tail = g[max_idx:]
    min_loc = int(np.argmin(tail))                    
    g_min = tail[min_loc]
    seg = slice(max_idx, max_idx + min_loc + 1)        
    min_loc_2 = None
    if min_loc - max_idx >= 2: 
        rebound_peaks = find_peaks(g[seg])[0]              
        if rebound_peaks.size:
            sub = g[max_idx: max_idx + rebound_peaks[0] + 1]   
            min_loc_2 = int(np.argmin(sub))
            g_min_2 = sub[min_loc_2]
            seg = slice(max_idx, max_idx + min_loc_2 + 1) 

    # skip if slope cannot be computed
    if max_idx == len(g) - 1 or g_max - g[0] <= 0.5 or (g < 3.5).any():
        continue
    if g_max == g_min:                                 
        misi_calc[s] = 0.0
        continue
  
    if min_loc_2 is not None and (g_min_2 - g_min > 0.5 or g[max_idx + min_loc_2 + 1] - g[max_idx + min_loc_2] > 0.5):
        continue
    slope = np.polyfit(np.arange(121)[seg], g[seg], 1)[0]
    misi_calc[s] = 1000 * abs(slope)/i_bar            

# gastric emptying
early_paracetamol_auc_0_30 = 15*para_cols[:, 1] + 7.5*para_cols[:, 2]

# Figure 2e
fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(rho.T, cmap='RdBu_r', vmin=-0.5, vmax=0.5) 
for r in range(6):
    for c in range(3):
        ax.annotate(f'{rho[c, r]:.2f}' + (f'\n{stars[c, r]}' if stars[c, r] else ''), xy=(c, r), ha='center', va='center', color='black', fontsize=12)

ax.set_xticks(range(3))
ax.set_xticklabels(['cA2', 'cD2', 'cD1'], fontsize=11)
ax.set_yticks(range(6))
ax.set_yticklabels(index_labels, fontsize=11)
ax.tick_params(axis='x', pad=22)                               

ax.scatter(range(3), [-0.02] * 3, transform=ax.get_xaxis_transform(), marker='s', s=144, c=['#5B2A86', '#D62728', '#FF7F0E'], linewidths=0.8, clip_on=False)


cbar = fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='6%'),
                    ticks=[-0.5, -0.25, 0, 0.25, 0.5])
cbar.set_label('Spearman correlations', fontsize=12)
plt.show()