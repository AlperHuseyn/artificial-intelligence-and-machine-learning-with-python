"""
This example shows how to use the softmax function to convert an array of
raw scores into probabilities. We have an array `scores` of length 10, which
contains the scores produced by a machine learning model for each digit (0-9)
based on input image. The `softmax` function is used to convert these scores
into probabilities by applying the softmax function element-wise to each score.
The resulting array of probabilities `probs` has the same length as `scores`
and represents the corresponding probability distribution over all possible outputs.
"""


import matplotlib.pyplot as plt
import numpy as np

# Define softmax function
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Example usage:
scores = np.array([9.2, 3.5, -0.1, 5.6, 7.8, 2.1, -1.5, 0.2, 4.9, 6.3])
probs = softmax(scores)

print(probs)