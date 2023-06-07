---

# Sampling from a Normal Distribution
- Draw a random sample of 100 elements from a normal distribution with a mean of 100 and standard deviation of 15.

## Creating a Confidence Interval
- Create a confidence interval for the population mean based on this sample with a 95% confidence level, assuming that the population standard deviation is unknown.

## Calculating Percentage of Means
- Select 1,000,000 random samples of 100 elements each from the normal distribution with a mean of 100 and standard deviation of 15.
- Calculate what percentage of the mean of these samples falls within the confidence interval you determined.

---

# Binomial Distribution
An important discrete distribution is called `binomial distribution`. The binomial distribution assumes that the result of a random experiment
is either of two outcomes, success or failure. If the probability of success in the random experiment is `p`, the probability of failure is `(1 – p)`.
If the random experiment is repeated `n times`, the probability of obtaining `k successful outcomes` can be shown as `P{X = k}`. The "probability mass function"
related to the binomial distribution is as follows:

```python
from math import comb

def binomial_distribution(n, k, p):
    return comb(n, k) * (p**k) * ((1-p)**(n-k))
```

You can see that the function depends on the values of n, k, and p. The first multiplicative term is the number of ways to choose k successes out of n trials.
Perform the following operations:

- Take `n = 20` and `p = 0.5` and let `k = 10` and calculate the probability value from the above formula.
- Compare the result with the `binom` singleton object in the `scipy.stats` module of the `SciPy library` as follows:

```python
from scipy.stats import binom
result = binom.pmf(k, n, p)
```

---

# Binomial Distribution's Probability Function Resembles Normal Distribution

The probability function of the Binomial Distribution approximates the normal distribution (i.e. when the graph of the values of `k` on the
horizontal axis and the values of `P{X = k}` on the vertical axis is plotted, a Gaussian curve is obtained, given a certain `n` and `p`). Show this
experimentally as follows:

- Obtain the probabilities of `k` from 0 to 1000 for `n` = 1000 and `p` = 0.5.

- Then, plot a graph using the `matplotlib.plot` function, with `k` values (from 0 to 1000) on the horizontal axis and probability values on the vertical axis, to check whether it resembles the normal distribution.

- Using the `binom` singleton object in the `scipy.stats` module, obtain 1000 random values of `k` with `n` = 100 and `p` = 0.5. Plot a histogram of these values and check visually if it resembles the normal distribution. Then, perform the Shapiro-Wilk test on these values to determine their conformity with the normal distribution.

----
