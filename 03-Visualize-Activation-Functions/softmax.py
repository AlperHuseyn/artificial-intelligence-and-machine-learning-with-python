import matplotlib.pyplot as plt
import numpy as np

# Define softmax function
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Generate 100 points between -5 and 5 along the x-axis
x = np.linspace(-10, 10, 100)

# Calculate corresponding y values using softmax function
y = softmax(x)

# Create plot with custom styling
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, linewidth=2, color='blue')

# Set plot title and axis labels
ax.set_title('Softmax Activation Function', fontsize=18)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)

# Customize tick labels and grid lines
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(alpha=0.3)

# Save the plot as a JPEG file
plt.savefig('softmax.jpg', dpi=300, bbox_inches='tight')

# Show plot
plt.show()
