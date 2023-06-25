import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dataloader import load_mouse_data
# import adam optimizer

X_train, X_test, y_train, y_test = load_mouse_data()
input_shape = (X_train.shape[1],)
# Create the model
model = Sequential()

# Add layers to the model
model.add(Dense(64, activation='relu', input_shape=input_shape))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer =Adam(learning_rate=0.001)
model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
