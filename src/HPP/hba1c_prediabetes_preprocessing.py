"""
Extract HbA1c and medical conditions for prediabetes from HPP dataset.

Input : df_swt.csv
Output: df_a1c.csv for 4_phenotyping_validation.py,
        df_hba1c.csv for 3_cross_level_analysis.py and 4_phenotyping_validation.py,
        df_prediab.csv for 4_phenotyping_validation.py
"""
import pandas as pd
from pheno_utils import PhenoLoader

# find t0 (first meal date of each CGM participant)
df_t0 = pd.read_csv('df_swt.csv')
df_t0 = pd.to_datetime(df_t0.groupby('participant_id')['meal_date'].min()).dt.normalize().rename('t0').reset_index()

# save all HbA1c measurements and compute gap in days from t0
pl_blood = PhenoLoader('blood_tests')
df_a1c = pl_blood[pl_blood.fields].reset_index()
df_a1c = df_a1c[df_a1c['participant_id'].isin(df_t0['participant_id'])]
df_a1c = (df_a1c.assign(a1c_date=pd.to_datetime(df_a1c['collection_date']).dt.normalize(),
                        a1c=pd.to_numeric(df_a1c['bt__hba1c_float_value'], errors='coerce'))
df_a1c = df_a1c.dropna(subset=['a1c_date', 'a1c'])
df_a1c = df_a1c.groupby(['participant_id', 'a1c_date'], as_index=False)['a1c'].mean()
df_a1c = df_a1c.merge(df_t0, on='participant_id'))
df_a1c['gap'] = (df_a1c['a1c_date'] - df_a1c['t0']).dt.days
df_a1c.to_csv('df_a1c.csv', index=False)

# find the draw closest to t0 within one year for cross-level analysis
df_hba1c = df_a1c.assign(abs_gap=df_a1c['gap'].abs()).sort_values(['abs_gap', 'gap']).drop_duplicates('participant_id')
df_hba1c = (df_hba1c[df_hba1c['abs_gap'] < 365].sort_values('participant_id').reset_index(drop=True))
df_hba1c.to_csv('df_hba1c.csv', index=False)

# find subjects with self-reported prediabetes
pl_med = PhenoLoader('medical_conditions')
df_medical = pl_med[pl_med.fields]
df_medical_initial = df_medical[(df_medical.index.get_level_values('research_stage') == '00_00_visit') & (df_medical['data_source'] == 'initial medical survey') & (df_medical['medical_condition'] == 'pre diabetes')].reset_index()
df_medical = df_hba1c.merge(df_medical, on='participant_id')

# define prediabetes as HbA1c > 5.7 before t0 (t0 day included) or self-reported prediabetes in the initial medical survey
df_prediab = (pd.concat([df_a1c[(df_a1c['gap'] >= -365) & (df_a1c['gap'] <= 0) & (df_a1c['a1c'] > 5.7)], df_medical])xw.sort_values('a1c', ascending=False).drop_duplicates('participant_id'))
df_prediab.to_csv('df_prediab.csv', index=False)
