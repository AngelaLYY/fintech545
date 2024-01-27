'''Given the dataset in problem1.csv:
a. calculate the first four moments values by using normalized formula in the "Week1 - UnivariateStats".
b. calculate the first four moments values again by using your chosen statistical package.
c. Is your statistical package functions biased? Prove or disprove your hypothesis respectively. 
Explain your conclusion.
'''

import pandas as pd
import numpy as np

# import package for part b and part c (test biasness)
from scipy.stats import skew, kurtosis, ttest_ind 



# Calculation of moments

def first4Moments(sample):
    n = len(sample)

    # mean
    μ_hat = sum(sample)/n

    # remove the mean from the sample
    sim_corrected = sample - μ_hat
    cm2 = np.dot(sim_corrected, sim_corrected) / n

    # variance (normalized)
    σ2_hat = np.dot(sim_corrected, sim_corrected) / (n-1)

    # skewness (normalized)
    skew_hat = np.sum(sim_corrected**3) / n / np.sqrt(cm2**3)

    # kurtosis (normalized)
    kurt_hat = np.sum(sim_corrected**4) / n / cm2**2

    excessKurt_hat = kurt_hat - 3

    return μ_hat, σ2_hat, skew_hat, excessKurt_hat


# parse the dataset from the csv file
dataset = pd.read_csv('problem1.csv')
sample = dataset['x']

# generated results using normalized formula
m, s2, sk, k = first4Moments(sample)

# generated results using packages numpy and scipy
pm, ps2, psk, pk = np.mean(sample), np.var(sample), skew(sample), kurtosis(sample)

# with open('output.txt', 'w') as f:
      

# use t-test for each moment 
t_statistic_mean, p_value_mean = ttest_ind(sample, np.random.normal(m, pm, len(sample)))
t_statistic_variance, p_value_variance = ttest_ind(sample, np.random.normal(s2, ps2, len(sample)))
t_statistic_skewness, p_value_skewness = ttest_ind(sample, np.random.normal(sk, psk, len(sample)))
t_statistic_kurtosis, p_value_kurtosis = ttest_ind(sample, np.random.normal(k, pk, len(sample)))

# set alpha for conclusion later
alpha = 0.05


# output test result 
with open('output.txt', 'a') as f:
      print(f"normalized formula (stastical package)", file=f)
      print(f"Mean {m} ({pm})", file=f)
      print(f"Variance {s2} ({ps2})", file=f)
      print(f"Skew {sk} ({psk})", file=f)
      print(f"Kurtosis {pk} ({kurtosis(sample)})", file=f)

      print(f"Mean diff = {m - pm}", file=f)
      print(f"Variance diff = {s2 - ps2}", file=f)
      print(f"Skewness diff = {sk - psk}", file=f)
      print(f"Kurtosis diff = {k - pk}", file=f)

      print("T-test results for each moment:", file=f)
      print(f"Mean: T-statistic = {t_statistic_mean}, p-value = {p_value_mean}", file=f)
      print(f"Variance: T-statistic = {t_statistic_variance}, p-value = {p_value_variance}", file=f)
      print(f"Skewness: T-statistic = {t_statistic_skewness}, p-value = {p_value_skewness}", file=f)
      print(f"Kurtosis: T-statistic = {t_statistic_kurtosis}, p-value = {p_value_kurtosis}", file=f)
      
      print("\nConclusion:", file=f)

      if p_value_mean < alpha:
            print(f"We reject null hypothesis and conclude that the difference is significant and thus package is biased for mean calculation", file = f)
      else:
            print(f"We fail to reject null hypothesis and conclude that the difference is not significant and thus package is unbiased for mean calculation", file = f)

      if p_value_variance < alpha:
            print(f"We reject null hypothesis and conclude that the difference is significant and thus package is biased for variance calculation", file = f)
      else:
            print(f"We fail to reject null hypothesis and conclude that the difference is not significant and thus package is unbiased for variance calculation", file = f)

      if p_value_skewness < alpha:
            print(f"We reject null hypothesis and conclude that the difference is significant and thus package is biased for skewness calculation", file = f)
      else:
            print(f"We fail to reject null hypothesis and conclude that the difference is not significant and thus package is unbiased for skewness calculation", file = f)

      if p_value_kurtosis < alpha:
            print(f"We reject null hypothesis and conclude that the difference is significant and thus package is biased for kurtosis calculation", file = f)
      else:
            print(f"We fail to reject null hypothesis and conclude that the difference is not significant and thus package is unbiased for kurtosis calculation", file = f)



# with open('output.txt', 'a') as f:
