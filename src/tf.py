import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout
from keras.optimizers import Adam
from dataloader import load_mouse_data
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

# import adam optimizer

X_train, X_test, y_train, y_test = load_mouse_data()
input_shape = (X_train.shape[1],)
# Create the model
model = Sequential()
num_labels = 39
hidden_units = 256
batch_size = 128
dropout = 0.45

# Add layers to the model
model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(40, activation='softmax'))
# Compile the model
optimizer =Adam(learning_rate=0.001)
model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
