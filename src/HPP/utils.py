"""Functions for mixed-effects fitting, model comparison and r2."""

import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def fit_mixedlm(formula, data, groups, re_formula="~carb_within"):
    """
    Fit a mixed-effects model with fallbacks after fitting exceptions:
    default -> lbfgs/500 iter -> nm/1000 iter -> random-intercept-only.
    Returns (fit, status); accepts a fit without checking fit.converged.
    """
    methods = [
        {'reml': True},
        {'reml': True, 'method': 'lbfgs', 'maxiter': 500},
        {'reml': True, 'method': 'nm', 'maxiter': 1000},
    ]
    for i, kwargs in enumerate(methods):
        try:
            fit = smf.mixedlm(formula, data=data, groups=groups, re_formula=re_formula).fit(**kwargs)
            return fit, "OK" if i == 0 else f"OK_fallback_{kwargs['method']}"
        except Exception as e:
            if i < len(methods) - 1:
                continue
            # last resort: random intercept only
            try:
                fit = smf.mixedlm(formula, data=data, groups=groups, re_formula="~1").fit(reml=True)
                return fit, "OK_intercept_only"
            except Exception:
                return None, f"FAILED: {str(e)}"


def cross_nutrient_mixedlm(formula, data, groups, re_formula=None):
    """Fit mixed model with convergence fallbacks. Returns (fit, status)."""
    md = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
    for kw, label in [
        (dict(reml=True),                               'OK'),
        (dict(reml=True, method='lbfgs',  maxiter=500), 'OK_lbfgs'),
        (dict(reml=True, method='nm',     maxiter=1000), 'OK_nm'),
        (dict(reml=True, method='powell', maxiter=1000), 'OK_powell'),
    ]:
        try:
            fit = md.fit(**kw)
            if fit.converged:
                return fit, label
        except Exception:
            pass
    # last resort: intercept only
    try:
        fit = smf.mixedlm(formula, data, groups=groups, re_formula="~1").fit(reml=True)
        return fit, 'INTERCEPT_ONLY'
    except Exception as e:
        return None, f'FAILED:{str(e)[:60]}'


def lrt_test(llf_full, llf_ref, n_extra=2):
    """Return (statistic, standard p-value, boundary-mixture p-value).

    Preserve the existing chi-square mixtures: 1/2 each for df=1,2 when
    n_extra=2; 1/3 each for df=3,4,5 when n_extra=5.
    """
    stat = max(2 * (llf_full - llf_ref), 0)
    p_std = 1 - chi2.cdf(stat, df=n_extra)
    if n_extra == 2:
        p_mix = 0.5*(1 - chi2.cdf(stat, 1)) + 0.5*(1 - chi2.cdf(stat, 2))
    elif n_extra == 5:
        p_mix = (1/3)*(1-chi2.cdf(stat,3)) + (1/3)*(1-chi2.cdf(stat,4)) + (1/3)*(1-chi2.cdf(stat,5))
    else:
        p_mix = p_std
    return stat, p_std, p_mix


def fit_joint_model(data, baseline, joint_interact, formula_nutrients, base_cols, min_meals=5):
    """Fit one response's Phase B model and return its results dictionary.

    Use complete records for the selected moderators and response, then retain
    participants with at least min_meals records. The caller selects moderators,
    including the response-specific gBLUP column, and handles result exports.
    """
    formula_rhs = (formula_nutrients + " + " + " + ".join(joint_interact)
                   + " + " + " + ".join(f"carb_within:{v}" for v in joint_interact))
    resp_col = baseline['resp_col']
    df_fit = data.dropna(subset=base_cols + joint_interact + [resp_col])
    meal_counts = df_fit.groupby('participant_id').size()
    df_fit = df_fit[df_fit['participant_id'].isin(meal_counts[meal_counts >= min_meals].index)].copy()

    formula = f"{resp_col} ~ {formula_rhs}"
    fit, status = fit_mixedlm(formula, df_fit, df_fit['participant_id'], "~carb_within")
    if fit is None:
        return {'status': 'FAILED', 'error': status}
    var_u1 = float(fit.cov_re.iloc[1, 1]) if fit.cov_re.shape[0] >= 2 else 0
    return {
        'status': status, 'fit': fit,
        'var_u0': float(fit.cov_re.iloc[0, 0]), 'var_u1': var_u1, 'var_e': fit.scale,
        'aic': fit.aic, 'n_obs': len(df_fit), 'n_groups': df_fit['participant_id'].nunique(),
        'pct_tau2_slope': 100 * var_u1 / baseline['var_u1'] if baseline['var_u1'] > 0 else 0,
        'fe_params': fit.fe_params.to_dict(), 'fe_pvalues': fit.pvalues.to_dict(),
        'formula': formula, 'joint_interact': joint_interact.copy(),
    }


def marginal_r2(result):
    """Return marginal and conditional R² for a random-intercept model."""
    var_fixed = np.var(result.model.exog @ result.fe_params.values, ddof=0)
    var_random = result.cov_re.values[0, 0]
    var_total = var_fixed + var_random + result.scale
    return var_fixed / var_total, (var_fixed + var_random) / var_total
