"""
Nested cross-validation endotype classificaition.

Input: initial_cohort_metabolic_phenotypes.csv (downloaded from the link under 'Data availability').
       dwt_features.csv (from 1_extract_features.py)
       conventional_features.csv (from 1_extract_features.py)
       frequency_features.csv  (from 1_extract_features.py)
       dwt_coefficients_level2_<wavelet>.csv (from 1_extract_features.py)

Outputs: 
    <target>_nested_cv_results.csv     Experiment results under each setting
    model_summary.csv                  descriptive statistics of model performance (target, extraction, classifier)
    Fig2b.png.       
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import nested_cv

parser = argparse.ArgumentParser()
parser.add_argument("--labels", nargs="+", default=["muscle_ir", "beta_cell", "incretin_effect", "hepatic_ir"])
parser.add_argument("--n_repeats", dest="n_repeats", type=int, default=100)
parser.add_argument("--n_jobs", dest="n_jobs", type=int, default=12)
parser.add_argument("--output_dir", dest="output_dir", type=Path, default=Path("results/nested_cv"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

# preprocess phenotype data
labels = pd.read_csv("initial_cohort_metabolic_phenotypes.csv").rename(columns={"SubjectID": "subject_id"})
labels = labels.sort_values("subject_id").reset_index(drop=True)

# match the features with labels
conventional = labels[["subject_id"]].merge(pd.read_csv("conventional_features.csv"), on="subject_id")
frequency = labels[["subject_id"]].merge(pd.read_csv("frequency_features.csv"), on="subject_id")
dwt_stats = pd.read_csv("dwt_features.csv")
matrices = {
    "extracted": conventional.drop(columns="subject_id").to_numpy(),
    "fft": frequency[["fft_max_amplitude", "fft_dominant_frequency", "fft75_frequency"]].to_numpy(),
    "welch_psd": frequency[["psd_max_amplitude"]].to_numpy(),
}
for wavelet in ["db2", "db4", "sym4", "coif1"]:
    stats = labels[["subject_id"]].merge(dwt_stats[dwt_stats["wavelet"] == wavelet], on="subject_id")
    coefficients = labels[["subject_id"]].merge(pd.read_csv(f"dwt_coefficients_level2_{wavelet}.csv"), on="subject_id")
    matrices[f"dwt_v3_{wavelet}"] = stats.loc[:, "cA2_mean":"cD1_entropy"].to_numpy()          
    matrices[f"dwt_v3b_{wavelet}"] = stats.loc[:, "global_mean":"global_entropy"].to_numpy()   
    matrices[f"dwt_v3b_new_{wavelet}"] = stats[["global_energy", "global_entropy"]].to_numpy()
    matrices[f"dwt_v4_{wavelet}"] = coefficients.filter(like="cA2_").to_numpy()                 

# nested CV for every experiment setting
metabolic_labels = { 
    "muscle_ir": ("MuscleIR_Class", "IR"),
    "beta_cell": ("BetaCellFunction_Class", "Dysfunction"),
    "incretin_effect": ("IE_Class", "Dysfunction"),
    "hepatic_ir": ("HepaticIR_Class", "IR"),
}

names, jobs = [], []
for target in args.labels:
    label_column, pos_label = metabolic_labels[target]
    for extraction, features in matrices.items():
        pca_components = None
        # v4 using pca on approximation coefficients
        if extraction.startswith("dwt_v4"):
            pca_components = 2         
        names.append((target, extraction))
        jobs.append(delayed(nested_cv)(features, labels[label_column].to_numpy(), pos_label,
                                       classifier_names=["L1_logistic", "L2_Logistic", "RBF_SVC", "RFR"],
                                       n_repeats=args.n_repeats, pca_components=pca_components))

parallel = Parallel(n_jobs=args.n_jobs, verbose=5)
results = parallel(jobs)
for (target, extraction), result in zip(names, results):
    result["target"] = target
    result["extraction"] = extraction
table = pd.concat(results)
for target in args.labels:
    rows = table[table["target"] == target]
    rows.to_csv(args.output_dir/f"{target}_nested_cv_results.csv", index=False)


# preocessing classification results
grouped = table.groupby(["target", "extraction", "classifier"], sort=False)["auroc"]
summary = grouped.agg(mean_auroc="mean", sd_auroc="std", n="count").reset_index()
summary[["version", "wavelet"]] = summary["extraction"].str.extract(r"dwt_(.+)_(.+)")   
summary.to_csv(args.output_dir/"model_summary.csv", index=False)                       #

# best model (highest mean AUROC) of each feature group for every label
summary["method"] = summary["extraction"].map({"extracted": "Extracted features", "fft": "FFT features",
                                               "welch_psd": "Welch PSD features"}).fillna("DWT features")
summary["ci95"] = 1.96*summary["sd_auroc"]/np.sqrt(summary["n"]) 
ranked = summary.sort_values(["mean_auroc", "sd_auroc"], ascending=[False, True])
best = ranked.drop_duplicates(["target", "method"])


# plot fig2b
x = np.arange(len(args.labels))
width = 0.18
colors = {"WT features": "#9E47BB", "Extracted features": "#2F7FB8",
          "FFT features": "#259B8D", "Welch PSD features": "#EAA22F"}
tick_labels = {"muscle_ir": "Muscle IR", "beta_cell": "Beta-Cell\nFunction",
               "incretin_effect": "Incretin\nEffect", "hepatic_ir": "Hepatic IR"}
fig, ax = plt.subplots(figsize=(10.4, 5.6))
for pos, method in zip(np.linspace(-1.5 * width, 1.5 * width, 4), colors):
    rows = best[best["method"] == method].set_index("target").loc[args.labels]
    ax.bar(x + pos, rows["mean_auroc"], width=width, yerr=rows["ci95"], capsize=3, label=method,
           color=colors[method], edgecolor="#0F172A", linewidth=0.6,
           error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#334155"})
ax.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=0.8)
ax.text(len(args.labels) - 0.18, 0.515, "AUROC = 0.5", fontsize=10, ha="right", va="bottom")
ax.set_xticks(x, [tick_labels[target] for target in args.labels])
ax.set_ylabel("Mean AUROC", fontsize=13)
ax.set_ylim(0.0, 1.05)
ax.tick_params(labelsize=11, color="#4B5563")
ax.grid(axis="y", alpha=0.22, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=11, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(args.output_dir/"Fig2b.png", dpi=300, bbox_inches="tight")


