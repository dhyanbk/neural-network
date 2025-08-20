import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

# Input range
x = np.linspace(-10, 10, 400)

# Compute activations
sig = sigmoid(x)
th = tanh(x)
re = relu(x)

# Softmax applied to 3 vectors: x, 0.5x, and 0.2x
softmax_input = np.vstack([x, 0.5 * x, 0.2 * x])
soft = softmax(softmax_input).T  # Shape: (400, 3)

# Plot all functions
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, sig, label='Sigmoid', color='blue')
plt.title("Sigmoid")
plt.grid(True)

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x, th, label='Tanh', color='orange')
plt.title("Tanh")
plt.grid(True)

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x, re, label='ReLU', color='green')
plt.title("ReLU")
plt.grid(True)

# Softmax
plt.subplot(2, 2, 4)
plt.plot(x, soft[:, 0], label='Set 1', linestyle='-')
plt.plot(x, soft[:, 1], label='Set 2', linestyle='--')
plt.plot(x, soft[:, 2], label='Set 3', linestyle=':')
plt.title("Softmax")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
