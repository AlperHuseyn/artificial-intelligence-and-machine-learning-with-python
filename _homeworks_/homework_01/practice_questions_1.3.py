""" My approach to practice question #1.3, binomial distribution """

from math import comb
from scipy.stats import binom

def binomial_distribution(n, k, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)

n = 20; k = 10; p = .5
print(f'Using binomial_distribution func: {binomial_distribution(n, k, p)}')
print(f"Using scipy.stats module's binom: {binom(n, p).pmf(k)}")
