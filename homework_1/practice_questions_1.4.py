""" My approach to practice question #1.3, binomial distribution to gauss istribution """

from scipy.stats import binom, shapiro
import matplotlib.pyplot as plt
import numpy as np

# Obtain probabilities of k from 0 to 1000 for n = 1000 and p = 0.5
n = 1000
p = .5
k = np.arange(0, 1001)
probs = [binom.pmf(i, n, p) for i in k]

# Plot the graph of k vs probabilities
plt.plot(k, probs)
plt.title('Binomial Distribution vs Normal Distribution')
plt.xlabel('k')
plt.ylabel('Probability')
plt.show()
 