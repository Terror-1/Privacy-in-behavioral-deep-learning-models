import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Bidirectional, GRU, Dense
from keras.models import Sequential
from dataloader import load_mouse_data

X_train, X_test, y_train, y_test = load_mouse_data()
training_length = len(X_train)
batch_size = 256
noise_multiplier = 2.1
epochs = 100
delta = training_length**(-3/2)
noise_multiplier_arr=[]
eps_arr=[]
for x in np.arange(0.6, 5.2, 0.1):
    noise_multiplier = x
    noise_multiplier_arr.append(x)
    eps,_ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(n=training_length, batch_size=batch_size, noise_multiplier=noise_multiplier, epochs=epochs, delta=delta)
    eps_arr.append(eps)


print(noise_multiplier_arr)
fig, ax = plt.subplots()
vertical_line_x = 0.8

ax.plot(noise_multiplier_arr,eps_arr)
ax.set_xlabel('Noise Multiplier')
ax.set_ylabel('Epsilon')
ax.set_title('Epsilon vs Noise Multiplier')
ax.axvline(vertical_line_x, color='red', linestyle='--', label='Vertical Line')
intersection_y = eps_arr[2]
ax.plot(vertical_line_x, intersection_y, marker='o', color='red', markersize=8)
plt.savefig('../results/noisevseps.png')

plt.show()
x_start = 0.2
x_end = 7

# Set the color and transparency of the highlighted region (adjust as needed)
highlight_color = 'lightgrey'
alpha = 0.3

# Use axvspan() to create the shaded background region
