import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import pandas as pd
from dataloader import load_mouse_data
tf.get_logger().setLevel('ERROR')

# Load the training and validation data
X_train, X_test, y_train, y_test = load_mouse_data()
train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

dpsgd = True
input_dim = 5
num_classes = 40
epochs = 1
batch_size = 64
training_size = 0.8
training_length = len(X_train)
testing_length = len(X_test)
l2_norm_clip = 1.0
noise_multiplier = 1.1
num_microbatches = 128
learning_rate = 0.001
delta = training_length**(-3/2)

combined_train_dataset = tf.data.Dataset.zip((train_dataset, train_labels_dataset)).batch(batch_size)
combined_test_dataset= tf.data.Dataset.zip((test_dataset, test_labels_dataset)).batch(batch_size)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
if dpsgd:
    optimizer = tensorflow_privacy.DPKerasAdamOptimizer(l2_norm_clip=l2_norm_clip,noise_multiplier=noise_multiplier,learning_rate=learning_rate)
else :
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
print("Train on {training_length} samples, validate on {testing_length} samples".format(training_length=training_length, testing_length=testing_length))
model.fit(X_train, y_train,
          epochs=epochs,
          validation_data=(X_test, y_test),
          batch_size=batch_size)

loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

privacy_report=compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(number_of_examples=training_length,
                                              batch_size=batch_size,
                                              noise_multiplier=0.8,
                                              num_epochs=epochs,
                                              delta=1e-5,
                                              used_microbatching=False)
eps,_ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(n=training_length, batch_size=batch_size, noise_multiplier=noise_multiplier, epochs=epochs, delta=delta)
print("The training is eps-delta private with (ε = {epsilon}, δ = {delta})".format(epsilon=round(eps,3), delta="{:e}".format(delta)))