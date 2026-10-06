# Temporal structure in postprandial glycemia
This repository contains code for the manuscript "Temporal structure in postprandial glycemia identifies personalized high-risk metabolic profiles". This study analyzed postprandial glucose responses from over 4,000 adults under free-living conditions using multiscale decomposition to characterize broad excursion patterns and finer fluctuations. The study also used two external datasets to evaluate multiscale temporal features for metabolic subtype classification in deeply phenotyped adults and examined associations between temporal glucose features and metabolic indices in young adults.

![](figures/figure1.png)

## Data availability

Access to the Human Phenotype Project (HPP) used in this study can be requested through [https://humanphenotypeproject.org/](https://humanphenotypeproject.org/). The Stanford OGTT dataset is available at [https://www.nature.com/articles/s41551-024-01311-6](https://www.nature.com/articles/s41551-024-01311-6). The COPSAC2000 dataset is available upon request at [mortenr@food.ku.dk](mailto:mortenr@food.ku.dk). 

We provide a synthetic dataset for testing the code in this repository. The synthetic dataset is available at `synthetic_data/synthetic_data.csv`. The synthetic dataset is generated to mimic the structure of the real datasets used in this study, but it does not contain any real participant data or physiological information. The synthetic dataset is generated using [simglucose](https://github.com/jxx123/simglucose).

## This repository contains code in `src` used for:

Main analyses on the HPP dataset:
- `1_wt_and_scalar_features.py` Wavelet feature and conventional metrics extraction from postprandial CGM curves.
- `2_mixed_effects_models.py`Progressive multilevel model building (null, nutrients, meal timing, random carbohydrate slopes, multi-nutrient random slopes) with variance component analysis.
- `3_cross_level_interaction.py`Cross-level interaction analysis: independent screening and joint moderator model.
- `4_phenotyping_validation.py`Probabilistic phenotype classification into four response profiles with cardiometabolic and pre-diabetes validation. Longitudinal association with HbA1c.

External validation on Stanford OGTT dataset: 
- `1_OGTT_features.py`Wavelet feature extraction from OGTT data.
- `2_nested_cv.py` Discrimination of mechanistically defined dysglycemia endotypes using nested cross-validation.
- `3_leave_one_component_out.py`Leave-one-component-out analysis to evaluate the contribution of each wavelet feature.

External validation on COPSAC dataset:
- `correlation_metabolic_indices.py`Wavelet feature extraction from standard mixed meal challenge data, computation of metabolic indices, and association between wavelet features and metabolic indices. 
