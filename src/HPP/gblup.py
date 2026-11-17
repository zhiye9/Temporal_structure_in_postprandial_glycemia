import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import zscore

gcta_path = '/gcta'
plink_base = '/plink_folder' 
grm_prefix = 'grm_all_unrelated'
input_dir = 'greml_inputs'
out_dir = 'greml_outputs'

qcovar_path = f'{plink_base}/qcovar.txt'
covar_path = f'{plink_base}/covar.txt'

os.makedirs(input_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

response_cols = [
    'swt_db2_energy_cA2_log',
    'swt_db2_energy_cD2_log',
    'swt_db2_energy_cD1_log',
    'iauc_log',
    'peak_rise_log',
    'peak_time',
]
# Output names correspond to response_cols in the same order.
gblup_cols = [
    'gblup_cA2',
    'gblup_cD2',
    'gblup_cD1',
    'gblup_iAUC',
    'gblup_peak_rise',
    'gblup_peak_time',
]

df_swt = pd.read_csv('df_swt.csv', dtype={'participant_id': str})

from pheno_utils import PhenoLoader

df_gene = PhenoLoader('human_genetics').dfs['human_genetics'].reset_index()
df_gene['participant_id'] = df_gene['participant_id'].astype(str)
df_gene_swt = df_gene[df_gene['participant_id'].isin(df_swt['participant_id'])].copy()

fam = pd.read_csv(
    f'{plink_base}/chr_1.fam',
    sep=r'\s+',
    header=None,
    usecols=[0, 1],
    names=['FID', 'IID'],
    dtype=str,
)
overlap = fam[fam['IID'].isin(df_gene_swt['participant_id'])].copy()
overlap[['FID', 'IID']].to_csv('keep_list.txt', sep='\t', header=False, index=False)

subprocess.run([
    gcta_path,
    '--mbfile', f'{plink_base}/mbfile.txt',
    '--keep', 'keep_list.txt',
    '--maf', '0.01',
    '--make-grm',
    '--out', 'grm_all',
    '--thread-num', '4',
], check=True)

subprocess.run([
    gcta_path,
    '--grm', 'grm_all',
    '--grm-cutoff', '0.05',
    '--make-grm',
    '--out', grm_prefix,
], check=True)

grm_ids = pd.read_csv(f'{grm_prefix}.grm.id', sep=r'\s+', header=None, names=['FID', 'IID'], dtype=str)

person_means = (df_swt.groupby('participant_id')[response_cols]
                     .mean().reset_index())
df_base = (grm_ids.merge(person_means, left_on='IID', right_on='participant_id',
                         how='inner')
                  .reset_index(drop=True))

pheno_tags = []
for col in response_cols:
    tag = f'{col}_zscore'
    df_pheno = df_base[['FID', 'IID', col]].dropna(subset=[col]).copy()
    sd = df_pheno[col].std(ddof=0)

    df_pheno['pheno'] = zscore(df_pheno[col].to_numpy(dtype=float), ddof=0)
    df_pheno[['FID', 'IID', 'pheno']].to_csv(
        f'{input_dir}/pheno_{tag}.txt', sep='\t', header=False, index=False
    )
    pheno_tags.append(tag)
    print(f"{tag}: N={len(df_pheno)}")

for tag in pheno_tags:
    subprocess.run([
        gcta_path, '--reml',
        '--grm', grm_prefix,
        '--pheno', f'{input_dir}/pheno_{tag}.txt',
        '--qcovar', qcovar_path,
        '--covar', covar_path,
        '--cvblup',
        '--thread-num', '4',
        '--out', f'{out_dir}/cvblup_{tag}',
    ], check=True, capture_output=True, text=True)

all_cvblups = df_base[['FID', 'IID', 'participant_id']].copy()
for col, gblup_col in zip(response_cols, gblup_cols):
    tag = f'{col}_zscore'
    if tag not in pheno_tags:
        continue
    cv = pd.read_csv(
        f'{out_dir}/cvblup_{tag}.indi.cvblp', sep=r'\s+', header=None,
        usecols=[0, 1, 3], names=['FID', 'IID', gblup_col],
        dtype={'FID': str, 'IID': str}
    )
    all_cvblups = all_cvblups.merge(
        cv, on=['FID', 'IID'], how='left', validate='one_to_one'
    )

all_cvblups = (all_cvblups.set_index('participant_id', verify_integrity=True)
                         .reindex(columns=gblup_cols).reset_index())
cvblup_csv = 'GCTA/greml_outputs_v2/all_cvblups_zscore_v2.csv'
os.makedirs(os.path.dirname(cvblup_csv), exist_ok=True)
all_cvblups.to_csv(cvblup_csv, index=False)