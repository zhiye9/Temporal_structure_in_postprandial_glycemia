"""
Leave-one-component-out nested CV for metabolic endotypes to quantify the contribution of temporal features.

Input: initial_cohort_metabolic_phenotypes.csv (downloaded from the link under 'Data availability').
       dwt_features.csv (from 1_extract_features.py)
       dwt_coefficients_level2_<wavelet>.csv (from 1_extract_features.py)
       model_summary.csv (from 2_nested_cv.py)
       <target>_nested_cv_results.csv (from 2_nested_cv.py)

Outputs:
    selected_configs.csv        chosen (version, wavelet, classifier) per target, its rank and the overall best
    <target>_drop_results.csv   AUROC per (dropped band, repeat, fold) with the chosen classifier
    contribution_values.csv     mean AUROC loss per removed band, paired by outer fold, with naive 95 % CI
    Fig2c.png
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
parser.add_argument("--nested_dir", dest="nested_dir", type=Path, default=Path("results/nested_cv"))
parser.add_argument("--output_dir", dest="output_dir", type=Path, default=Path("results/subband_ablation"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

# rank the DWT configurations of every label, each with its best classifier
summary = pd.read_csv(args.nested_dir/"model_summary.csv")
summary = summary[summary["version"].notna()]
ranked = summary.sort_values(["mean_auroc", "sd_auroc", "version", "wavelet", "classifier"],
                             ascending=[False, True, True, True, True])
ranked = ranked.drop_duplicates(["target", "extraction"])
ranked["rank"] = ranked.groupby("target").cumcount() + 1
ranked["overall_best"] = ranked["version"] + "/" + ranked["wavelet"] + "/" + ranked["classifier"]
ranked["overall_best"] = ranked.groupby("target")["overall_best"].transform("first")

# best configuration
selected = ranked[ranked["version"] != "v4"].drop_duplicates("target")
selected = selected.set_index("target").loc[args.labels].reset_index()
selected.to_csv(args.output_dir/"selected_configs.csv", index=False)

# preprocess phenotype data
labels = pd.read_csv("initial_cohort_metabolic_phenotypes.csv").rename(columns={"SubjectID": "subject_id"})
labels = labels.sort_values("subject_id").reset_index(drop=True)
dwt_stats = pd.read_csv("dwt_features.csv")
metabolic_labels = {
    "muscle_ir": ("MuscleIR_Class", "IR"),
    "beta_cell": ("BetaCellFunction_Class", "Dysfunction"),
    "incretin_effect": ("IE_Class", "Dysfunction"),
    "hepatic_ir": ("HepaticIR_Class", "IR"),
}

# nested CV of the selected settings with one component removed
names, jobs = [], []
for config in selected.itertuples():
    label_column, pos_label = metabolic_labels[config.target]
    stats = labels[["subject_id"]].merge(dwt_stats[dwt_stats["wavelet"] == config.wavelet], on="subject_id")
    stats = stats.loc[:, "cA2_mean":"cD1_entropy"]
    coefficients = labels[["subject_id"]].merge(pd.read_csv(f"dwt_coefficients_level2_{config.wavelet}.csv"), on="subject_id")
    coefficients = coefficients.drop(columns="subject_id")
    for band in ["cA2", "cD2", "cD1"]:
        c = coefficients.loc[:, ~coefficients.columns.str.startswith(band)].to_numpy()
        p = np.abs(c) / np.abs(c).sum(axis=1, keepdims=True)
        energy = (c ** 2).sum(axis=1)
        entropy = -(p * np.log2(p + 1e-12)).sum(axis=1)

        if config.version == "v3":
            features = stats.loc[:, ~stats.columns.str.startswith(band)].to_numpy()
        elif config.version == "v3b":
            features = np.column_stack([c.mean(axis=1), c.std(axis=1), energy, entropy])
        else:
            features = np.column_stack([energy, entropy])
        names.append((config.target, band))
        jobs.append(delayed(nested_cv)(features, labels[label_column].to_numpy(), pos_label,
                                       classifier_names=[config.classifier], n_repeats=args.n_repeats))

parallel = Parallel(n_jobs=args.n_jobs, verbose=5)
results = parallel(jobs)
for (target, band), result in zip(names, results):
    result["target"] = target
    result["band"] = band
table = pd.concat(results)
for target in args.labels:
    rows = table[table["target"] == target]
    rows.to_csv(args.output_dir/f"{target}_drop_results.csv", index=False)

# pair every fold of the ablated model with the same fold of the full model
full = []
for config in selected.itertuples():
    rows = pd.read_csv(args.nested_dir/f"{config.target}_nested_cv_results.csv")
    full.append(rows[(rows["extraction"] == config.extraction) & (rows["classifier"] == config.classifier)])
full = pd.concat(full)
paired = full.merge(table, on=["target", "repeat", "fold"], suffixes=("_full", "_drop"))
paired["delta_auroc"] = paired["auroc_full"] - paired["auroc_drop"]
paired["hyperparameter_changed"] = paired["hyperparameter_full"] != paired["hyperparameter_drop"]

# AUROC loss upon removal of each component
grouped = paired.groupby(["target", "band"], sort=False)
contributions = grouped.agg(full_mean_auroc=("auroc_full", "mean"), drop_mean_auroc=("auroc_drop", "mean"),
                            mean_delta_auroc=("delta_auroc", "mean"), sd_delta_auroc=("delta_auroc", "std"),
                            n=("delta_auroc", "count"),
                            prop_hyperparameter_changed=("hyperparameter_changed", "mean")).reset_index()
contributions["ci95"] = 1.96*contributions["sd_delta_auroc"]/np.sqrt(contributions["n"])
contributions.to_csv(args.output_dir/"contribution_values.csv", index=False)


# plot fig2c
x = np.arange(len(args.labels))
width = 0.22
colors = {"cA2": "#5B2C6F", "cD2": "#d62728", "cD1": "#ff7f0e"}
tick_labels = {"muscle_ir": "Muscle IR", "beta_cell": "Beta-Cell Function",
               "incretin_effect": "Incretin Effect", "hepatic_ir": "Hepatic IR"}
fig, ax = plt.subplots(figsize=(10.4, 5.6))
for pos, band in zip(np.linspace(-width, width, 3), colors):
    rows = contributions[contributions["band"] == band].set_index("target").loc[args.labels]
    ax.bar(x + pos, rows["mean_delta_auroc"], width=width, yerr=rows["ci95"], capsize=3, label=band,
           color=colors[band], edgecolor="#0F172A", linewidth=0.6,
           error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#334155"})
ax.axhline(0, color="#4B5563", linewidth=0.8)
ax.set_xticks(x, [tick_labels[target] for target in args.labels])
ax.set_ylabel("AUROC loss upon component removal", fontsize=13)
lower = contributions["mean_delta_auroc"] - contributions["ci95"]
upper = contributions["mean_delta_auroc"] + contributions["ci95"]
ax.set_ylim(min(0.0, lower.min()*1.10), max(upper.max()*1.10, 0.05))
ax.tick_params(labelsize=11, color="#4B5563")
ax.grid(axis="y", alpha=0.22, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=11, frameon=False)
fig.tight_layout()
fig.savefig(args.output_dir/"Fig2c.png", dpi=300, bbox_inches="tight")