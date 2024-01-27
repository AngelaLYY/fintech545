# pip install statsmodels

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize

def myll(params, x, y):
    beta = params[:-1]
    s = params[-1]
    n = len(y)
    e = y - np.dot(x, beta)
    s2 = s**2
    ll = -n/2 * np.log(s2 * 2 * np.pi) - np.dot(e, e) / (2 * s2)
    return -ll

# Initial guess for parameters
initial_params = np.ones(x.shape[1] + 1)

# MLE Optimization problem
result = minimize(myll, initial_params, args=(x, y), method='Nelder-Mead')

# Extracting results
estimated_params = result.x[:-1]
estimated_sigma = result.x[-1]

print("Estimated Betas in MLE:", estimated_params)
print("Estimated Sigma in MLE:", estimated_sigma)

# OLS estimation
model_ols = sm.OLS(y, x)
result_ols = model_ols.fit()
estimated_params_ols = result_ols.params
estimated_sigma_ols = np.std(result_ols.resid)
print("Estimated Betas in OLS:", estimated_params_ols)
print("Estimated Sigma in OLS:", estimated_sigma_ols)