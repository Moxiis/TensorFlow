import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


plt.imshow(x_train[5555], cmap="gray") # Import the image
plt.show() # Plot the image
print(df)
