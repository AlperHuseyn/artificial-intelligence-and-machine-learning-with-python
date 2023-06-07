""" My approach to practice question #1.1, basic statistics """

import numpy as np
from scipy.stats import norm

"""
# when sample_size is < 30 t-distribution being used
def confidence_level(mean, std, sample_size, alpha=.05):
    t_value = norm.ppf(1 - alpha / 2, sample_size - 1)
    lower_bound = mean - t_value * std / np.sqrt(sample_size)
    upper_bound = mean + t_value * std / np.sqrt(sample_size)
    
    return lower_bound, upper_bound
"""

# when sample_size is >= 30 normal distribution being used
def confidence_level(mean, std, sample_size, alpha=.05):
    z_value = norm.ppf(1 - alpha / 2)
    lower_bound = mean - z_value * std / np.sqrt(sample_size)
    upper_bound = mean + z_value * std / np.sqrt(sample_size)
    
    return lower_bound, upper_bound

def percentage_within_interval(mean, std, sample_size, num_samples, alpha=.05):
    lower, upper = confidence_level(mean, std, sample_size)
    count = 0
    for _ in range(num_samples):
        sample = np.random.normal(mean, std, sample_size)
        sample_mean = np.mean(sample)
        if lower <= sample_mean <= upper:
            count += 1
            
    return count / num_samples * 100


random_sample = norm.rvs(100, 15, 100)
mean = np.mean(random_sample)
std = np.std(random_sample)
sample_size = 100
num_samples = 1_000_000

result = percentage_within_interval(mean, std, sample_size, num_samples)
print(f'{result:.2f}%')
 