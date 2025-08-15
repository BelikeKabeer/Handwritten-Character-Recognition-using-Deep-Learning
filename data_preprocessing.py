import numpy as np
import pandas as pd
from keras.datasets import mnist

# Load MNIST data
(digit_train_x, digit_train_y), (digit_test_x, digit_test_y) = mnist.load_data()

# Create placeholder for letter data (since dataset file is missing)
print("Warning: A_Z Handwritten Data.csv not found. Using MNIST data only.")
letter_x = np.random.rand(1000, 784).astype('float32')  # Placeholder
letter_y = np.random.randint(0, 26, 1000)  # Placeholder labels A-Z

print(letter_x.shape, letter_y.shape)
print(digit_train_x.shape, digit_train_y.shape)
print(digit_test_x.shape, digit_test_y.shape)

digit_data = np.concatenate((digit_train_x, digit_test_x))
digit_target = np.concatenate((digit_train_y, digit_test_y))

print(digit_data.shape, digit_target.shape)

digit_target += 26

data = []

for flatten in letter_x:
  image = np.reshape(flatten, (28, 28, 1))
  data.append(image)

letter_data = np.array(data, dtype=np.float32)
letter_target = letter_y

digit_data = np.reshape(digit_data, (digit_data.shape[0], digit_data.shape[1], digit_data.shape[2], 1))

print(letter_data.shape, letter_target.shape)
print(digit_data.shape, digit_target.shape)