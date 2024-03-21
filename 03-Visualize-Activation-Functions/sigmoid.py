import matplotlib.pyplot as plt
import numpy as np

# Define Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

# Generate 100 points between -5 and 5 along the x-axis
x = np.linspace(-5, 5, 100)

# Calculate corresponding y values using sigmoid function
y = sigmoid(x)

# Create plot with custom styling
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, linewidth=2, color='blue')

# Set plot title and axis labels
ax.set_title('Sigmoid Activation Function', fontsize=18)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)

# Customize tick labels and grid lines
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(alpha=0.3)

# Save the plot as a JPEG file
plt.savefig('sigmoid.jpg', dpi=300, bbox_inches='tight')

# Show plot
plt.show()
