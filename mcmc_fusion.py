"""
MCMC Bayesian Sensor Fusion Module.
File: mcmc_fusion.py

Standalone module for Bayesian sensor fusion of R-TFT and L-TFT predictions
using NUTS MCMC with Student-t distributions.

Prior = L-TFT (lagged/temporal) predictions
Likelihood = R-TFT (recent/main) predictions

Usage:
    import mcmc_fusion
    combined, lower, upper, rhats, ess = mcmc_fusion.combine_predictions(
        pred_5th_main, pred_95th_main, pred_50th_main,
        pred_5th_temporal, pred_95th_temporal, pred_50th_temporal
    )
"""

import numpy as np
import pymc as pm
import arviz as az
from joblib import Parallel, delayed
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
STUDENT_T_DF = 4
DRAWS = 1000
TUNE = 500
CHAINS = 4
CORES = 1
TARGET_ACCEPT = 0.8


# =============================================================================
# Core Functions
# =============================================================================
def estimate_uncertainty_from_quantiles(q5, q95, eps=1e-3):
    """Estimate sigma from 5th and 95th quantile predictions."""
    uncertainty = (q95 - q5) / (2 * 1.645)
    return np.where(uncertainty < eps, eps, uncertainty)


def bayesian_update(y_long, y_short, sigma_prior, sigma_likelihood, df=STUDENT_T_DF):
    """
    Bayesian sensor fusion using NUTS MCMC with Student-t distributions.
    
    Prior: temporal (long-term) prediction
    Likelihood: main (direct/short-term) prediction
    """
    with pm.Model() as model:
        mu_prior = pm.StudentT('mu', nu=df, mu=y_long, sigma=sigma_prior)
        pm.StudentT('y', nu=df, mu=mu_prior, sigma=sigma_likelihood, observed=y_short)
        
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            return_inferencedata=True
        )
    
    summary = az.summary(idata)
    rhat_value = summary['r_hat'].values[0]
    ess_bulk_value = summary['ess_bulk'].values[0]
    
    return idata, rhat_value, ess_bulk_value


def process_timestep(args):
    """Process a single datapoint-timestep pair."""
    y_long, y_short, q5_long, q95_long, q5_short, q95_short = args
    
    sigma_prior = estimate_uncertainty_from_quantiles(q5_long, q95_long)
    sigma_likelihood = estimate_uncertainty_from_quantiles(q5_short, q95_short)
    
    idata, rhat_value, ess_bulk_value = bayesian_update(
        y_long, y_short, sigma_prior, sigma_likelihood
    )
    
    posterior_samples = idata.posterior['mu'].values.flatten()
    return (
        np.mean(posterior_samples),
        np.percentile(posterior_samples, 5),
        np.percentile(posterior_samples, 95),
        rhat_value,
        ess_bulk_value
    )


def combine_predictions(pred_5th_main, pred_95th_main, pred_50th_main,
                        pred_5th_temporal, pred_95th_temporal, pred_50th_temporal):
    """
    Run MCMC sensor fusion across all datapoints and timesteps.
    
    Prior = temporal predictions, Likelihood = main predictions.
    
    Args:
        pred_5th_main, pred_95th_main, pred_50th_main: R-TFT quantile predictions
        pred_5th_temporal, pred_95th_temporal, pred_50th_temporal: L-TFT quantile predictions
    
    Returns:
        combined_predictions: posterior mean (n_groups, n_timesteps)
        lower_bound_90: 5th percentile of posterior (n_groups, n_timesteps)
        upper_bound_90: 95th percentile of posterior (n_groups, n_timesteps)
        rhat_values: list of R-hat convergence diagnostics
        ess_bulk_values: list of ESS bulk values
    """
    n_groups, n_timesteps = pred_5th_main.shape
    
    results = Parallel(n_jobs=-1)(
        delayed(process_timestep)(
            (pred_50th_temporal[i, j],
             pred_50th_main[i, j],
             pred_5th_temporal[i, j],
             pred_95th_temporal[i, j],
             pred_5th_main[i, j],
             pred_95th_main[i, j])
        )
        for i, j in tqdm(
            [(i, j) for i in range(n_groups) for j in range(n_timesteps)],
            total=n_groups * n_timesteps,
            desc="MCMC Fusion"
        )
    )
    
    combined_results = np.array([r[:3] for r in results]).reshape(n_groups, n_timesteps, 3)
    rhat_values = [r[3] for r in results]
    ess_bulk_values = [r[4] for r in results]
    
    return (
        combined_results[:, :, 0],  # combined_predictions (posterior mean)
        combined_results[:, :, 1],  # lower_bound_90 (5th percentile)
        combined_results[:, :, 2],  # upper_bound_90 (95th percentile)
        rhat_values,
        ess_bulk_values
    )
