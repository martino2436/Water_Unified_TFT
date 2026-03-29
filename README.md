# Unified Framework for Water Demand Forecasting and Early Leakage Warning

## Paper

**A Unified Framework for Water Demand Forecasting and Early Leakage Warning Using Uncertainty-Aware Temporal Fusion Transformers**

Martin Rapp, Jawad Fayaz

## Overview

This package provides a reproducible demo of the Combined Temporal Fusion Transformer (C-TFT) framework presented in the paper. The C-TFT fuses predictions from two base models — the Recent TFT (R-TFT, using 7 days of preceding flow) and the Lagged TFT (L-TFT, using flow from days 21–14 before the target) — through Bayesian MCMC sensor fusion to produce posterior predictions with calibrated uncertainty bounds.

The demo runs the full pipeline on 10 exemplar samples (5 leakage, 5 non-leakage) drawn from the held-out balanced test set:

1. Load trained R-TFT and L-TFT models
2. Generate predictions (regression + classification)
3. Bayesian MCMC sensor fusion → C-TFT posterior predictions
4. Denormalisation and forecast visualisation
5. Interpretability analysis (variable selection weights + temporal attention)

## Model Variants

|Model|Description|
|-|-|
|R-TFT (Recent)|7-day input → 24h forecast + leakage classification|
|L-TFT (Lagged)|Days 21–14 input → 24h forecast (regression only)|
|C-TFT (Combined)|MCMC Bayesian fusion of R-TFT + L-TFT|

## Folder Structure

```
Water_Unified_TFT/
├── Run_unified_framework.ipynb    # Main demo notebook
├── select_exemplar_samples.ipynb  # Sample selection tool (run once)
├── mcmc_fusion.py                 # Bayesian sensor fusion module
├── README.md
├── models/
│   ├── tft_weights.h5             # R-TFT trained weights
│   ├── tft_config.pkl             # R-TFT model configuration
│   ├── tft_temporal_weights.h5    # L-TFT trained weights
│   └── tft_temporal_config.pkl    # L-TFT model configuration
├── source/
│   ├── tft.py                     # R-TFT architecture
│   ├── tft_temporal.py            # L-TFT architecture
│   ├── tft_losses.py              # R-TFT custom losses and metrics
│   └── tft_losses_temporal.py     # L-TFT custom losses and metrics
├── data/
│   ├── exemplar_samples.pkl       # 10 exemplar samples (created by picker)
│   └── scalers.pkl                # Output scaler for denormalisation
└── figures/                       # Generated output figures
```

## Requirements

* Python 3.10+
* TensorFlow 2.x
* PyMC 5.x
* ArviZ
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* tqdm
* Statsmodels

Install with:

```bash
pip install tensorflow pymc arviz numpy pandas matplotlib seaborn scikit-learn joblib tqdm statsmodels
```

## How to Run



**1. \*\*Ensure all model files and data are in place\*\* as per the folder structure above.**



**2. \*\*Run the demo\*\*:**

&#x20;  **- Open `Run\_unified\_framework.ipynb`**

&#x20;  **- Run all cells sequentially**

&#x20;  **- The MCMC fusion step processes 960 operations (10 samples × 96 timesteps) and typically completes in a few minutes**


**3. \*\*Output figures\*\* are saved to the `figures/` directory.**MCMC Configuration

The Bayesian sensor fusion uses the No-U-Turn Sampler (NUTS) with the following configuration:

* 1000 posterior draws
* 500 tuning steps
* 4 independent chains
* Student-t distribution with 4 degrees of freedom
* Target acceptance rate: 0.8

## Data

The exemplar samples are drawn from the balanced test set (Test 1) of approximately 18,600 flow groupings from \~2,000 district metered areas (DMAs) in the UK, recorded at 15-minute intervals. Each sample contains 672 input timesteps (7 days) and 96 output timesteps (24 hours) of water flow, along with 5 known temporal features (weekday/weekend, hour of day, day of week, night/day, season) and 2 static features (DMA ID, pipe diameter).

