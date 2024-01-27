import numpy as np
import pandas as pd
import statsmodel.api as sm
from scipy.stats import norm

# read data from csv 
data = pd.read_csv('problem2.csv')

# define variables (converting to list)
x = data['x'].tolist()
y = data['y'].tolist()

# add the constant term
x = sm.add_constant(x)

# perform the regression and fit the model
result = sm.OLS(y,x).fit()

# print the summary table
print(result.sumary())


# Sample a random normal N(1.0, 5.0)
samples = 100
np.random.seed(0)  # for reproducibility
x = np.random.normal(1.0, 5.0, samples)


def myll(m, s):
    n = len(x)
    xm = x - m
    s2 = s**2
    ll = -n/2 * np.log(s2 * 2 * np.pi) - np.dot(xm, xm)/(2*s2)
    return ll


# Example usage
m = 2.0
s = 4.0
print(myll(m, s))
