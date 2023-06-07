""" My approach to practice question #1.2, bootstrap sampling """

import numpy as np
from scipy.stats import norm
import random

POPULATION_SIZE = 1_000_000
SAMPLE_SIZE = 1000
BOOTSTRAP_REPITITION = 1000

population = [random.randint(0, 1_000_000_000) for _ in range(POPULATION_SIZE)]
sample = [random.choice(population) for _ in range(SAMPLE_SIZE)]

# bootstrap sampling
means = []
for _ in range(BOOTSTRAP_REPITITION):
        bootstrap_samples = random.choices(sample, k=800)
        means.append(np.mean(bootstrap_samples))
        
lower_bootstrap = np.quantile(means, .025)
upper_bootstrap = np.quantile(means, .975)

# Basic statistics
lower_stats = np.mean(sample) - norm.ppf(.975) * np.std(sample) / np.sqrt(SAMPLE_SIZE)
upper_stats = np.mean(sample) + norm.ppf(.975) * np.std(sample) / np.sqrt(SAMPLE_SIZE)

population_mean = np.mean(population)

print(f'population mean: {population_mean:.2f}')
print(f'confidence bootstrap; lower: {lower_bootstrap:.2f}, upper: {upper_bootstrap:.2f}')
print(f'confidence basic stats; lower: {lower_stats:.2f}, upper: {upper_stats:.2f}')
