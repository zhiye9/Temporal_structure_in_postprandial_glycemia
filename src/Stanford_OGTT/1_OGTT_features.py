"""
Extract conventional, alternative frequency and DWT features from the OGTT glucose curves of the initial venous cohort.

Input: all_cohort_metabolicsubphenotyping_ogtt_glucose_09102023.csv (downloaded from the link under 'Data availability').

Output:
    handcrafted_features.csv                16 conventional OGTT features
    frequency_features.csv                  FFT and Welch PSD
    dwt_coefficients_level2_<wavelet>.csv   DWT coefficients
    dwt_features.csv                        DWT features (per-compoent/global mean, std, energy, entropy)
"""
import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import UnivariateSpline
from scipy.signal import welch

# read data
cgm = pd.read_csv("all_cohort_metabolicsubphenotyping_ogtt_glucose_09102023.csv")
cgm = cgm[(cgm["exp_type"] == "venous_without_matching_cgm_and_without_planned_athome_cgm") & (cgm["sample_location_extraction_method"] == "CTRU_Venous")]

# linear interpolation for conventional features
cgm_wide = cgm.pivot_table(index="timepoint", columns="subject_id", values="glucose", aggfunc="mean").sort_index()
cgm_imputed = cgm_wide.interpolate(method="linear", limit_direction="both", axis=0)
subjects = list(cgm_imputed.columns)     
minutes = cgm_imputed.index.to_numpy(dtype=float)

# conventional features 
cgm_curve = cgm_imputed.loc[cgm_imputed.index >= 0]   # 15 timepoints x 32 subjects
t = cgm_curve.index.to_numpy(dtype=float)
cgm_values = cgm_curve.to_numpy(dtype=float)
fpg = cgm_values[0]
peak = cgm_values.max(axis=0)
t_peak = t[cgm_values.argmax(axis=0)]         # time of the first maximum
below = (t[:, None] > t_peak) & (cgm_values < fpg)     # post-peak samples under the fasting value

conv_features = pd.DataFrame({"subject_id": subjects})
conv_features["ogtt_fpg"] = fpg
conv_features["ogtt_60"] = cgm_curve.loc[60].to_numpy()
conv_features["ogtt_120"] = cgm_curve.loc[120].to_numpy()
conv_features["ogtt_180"] = cgm_curve.loc[180].to_numpy()
conv_features["ogtt_auc"] = np.trapezoid(cgm_values, x=t, axis=0)
conv_features["ogtt_iauc"] = conv_features["ogtt_auc"] - 180 * fpg
conv_features["ogtt_pauc"] = np.trapezoid(np.clip(cgm_values - fpg, 0, None), x=t, axis=0)
conv_features["ogtt_nauc"] = conv_features["ogtt_pauc"] - conv_features["ogtt_iauc"]
conv_features["ogtt_max"] = peak
conv_features["ogtt_curve_size"] = np.abs(np.diff(cgm_values, axis=0)).sum(axis=0)
conv_features["ogtt_cv"] = cgm_values.std(axis=0, ddof=1) / cgm_values.mean(axis=0) * 100
conv_features["ogtt_time_baseline_peak"] = t_peak
conv_features["ogtt_time_peak_baseline"] = np.where(below.any(axis=0), t[below.argmax(axis=0)] - t_peak, 180.0)
conv_features["ogtt_slope_baseline_peak"] = np.divide(peak - fpg, t_peak, out=np.full(len(subjects), np.nan), where=t_peak > 0)
conv_features["ogtt_slope_peak_last"] = np.divide(cgm_values[-1] - peak, t[-1] - t_peak, out=np.full(len(subjects), np.nan), where=t_peak < t[-1])
conv_features["ogtt_time_below_basline"] = below.any(axis=0).astype(int)
conv_features.to_csv("conventional_features.csv", index=False)

# normalization and smoothing spline
cgm_zscore = (cgm_imputed - cgm_imputed.mean(axis=0))/cgm_imputed.std(axis=0, ddof=1)
cgm_smooth_curves = []
for subject in subjects:
    spline = UnivariateSpline(minutes, cgm_zscore[subject].to_numpy(), s=0.35*len(minutes))
    cgm_smooth_curves.append(spline(minutes))
cgm_smooth_curves = np.array(cgm_smooth_curves) 
cgm_centred = cgm_smooth_curves - cgm_smooth_curves.mean(axis=1, keepdims=True)    

# alternative frequency features
cgm_freq = np.arange(1, len(minutes)//2 + 1)/len(minutes)                
cgm_magnitude = np.abs(np.fft.rfft(cgm_centred, axis=1))[:, 1:]            
_, psd = welch(cgm_centred, fs=1.0, nperseg=8, noverlap=4, window="hann", scaling="density", return_onesided=True, detrend=False, axis=1)              
cgm_frequency = pd.DataFrame({"subject_id": subjects})
cgm_frequency["fft_max_amplitude"] = cgm_magnitude.max(axis=1)
cgm_frequency["fft_dominant_frequency"] = cgm_freq[cgm_magnitude.argmax(axis=1)]
cgm_frequency["fft75_frequency"] = [cgm_freq[np.searchsorted(np.cumsum(m), 0.75 * m.sum(), side="left")] for m in cgm_magnitude]                      
cgm_frequency["psd_max_amplitude"] = psd[:, 1:].max(axis=1)    
cgm_frequency.to_csv("frequency_features.csv", index=False)

# DWT features
cgm_dwt = []
for wavelet in ["db2", "db4", "sym4", "coif1"]:
    coefficients = [pywt.wavedec(c, wavelet, level=2) for c in cgm_centred]
    bands = {band: np.array([c[i] for c in coefficients]) for i, band in enumerate(["cA2", "cD2", "cD1"])}
    dwt_coeff = pd.DataFrame({"subject_id": subjects})
    for band, c in bands.items():
        dwt_coeff[[f"{band}_{k}" for k in range(c.shape[1])]] = c
    dwt_coeff.to_csv(f"dwt_coefficients_level2_{wavelet}.csv", index=False)

    stats = pd.DataFrame({"subject_id": subjects, "wavelet": wavelet})
    for name, c in list(bands.items()) + [("global", np.hstack(list(bands.values())))]:
        p = np.abs(c) / np.abs(c).sum(axis=1, keepdims=True)
        stats[f"{name}_mean"] = c.mean(axis=1)
        stats[f"{name}_std"] = c.std(axis=1)
        stats[f"{name}_energy"] = (c ** 2).sum(axis=1)
        stats[f"{name}_entropy"] = -(p * np.log2(p + 1e-12)).sum(axis=1)
    cgm_dwt.append(stats)
cgm_DWT_df = pd.concat(cgm_dwt)

cgm_DWT_df.to_csv("dwt_features.csv", index=False)