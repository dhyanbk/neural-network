import numpy as np
import matplotlib.pyplot as plt

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

# Input values
x = np.linspace(-10, 10, 400)

# Compute activations
y = {
    'sigmoid': 1 / (1 + np.exp(-x)),
    'tanh': np.tanh(x),
    'relu': np.maximum(0, x),
    'softmax': softmax(np.array([x, 0.5 * x, 0.2 * x])).T  # shape (400, 3)
}

# Plotting setup
plt.figure(figsize=(12, 8))
titles = ["Sigmoid", "Tanh", "ReLU", "Softmax"]
colors = ['blue', 'orange', 'green', 'purple']
linestyles = ['-', '--', ':']  # For softmax components

# Loop through and plot each activation
for i, (key, y_values) in enumerate(y.items()):
    plt.subplot(2, 2, i + 1)
    if key == 'softmax':
        for j in range(y_values.shape[1]):
            plt.plot(x, y_values[:, j], label=f'Softmax set {j+1}', linestyle=linestyles[j])
    else:
        plt.plot(x, y_values, label=key, color=colors[i])
    
    plt.title(f"{titles[i]} Activation Function")
    plt.xlabel("x")
    plt.ylabel("Activation")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
