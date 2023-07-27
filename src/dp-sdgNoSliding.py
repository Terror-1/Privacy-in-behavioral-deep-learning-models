import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import load_mouse_data
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

# Load the training and validation data
X_train, X_test, y_train, y_test = load_mouse_data()


dpsgd = False
input_dim = X_train.shape[1]
num_classes = 6
epochs = 100
batch_size = 128
training_size = 0.8
training_length = len(X_train)
testing_length = len(X_test)
l2_norm_clip = 1.1
noise_multiplier = 0.6
num_microbatches = 8
learning_rate = 0.001
delta = training_length**(-3/2)
training_length = ((training_length)-(training_length%num_microbatches))
testing_length = ((testing_length)-(testing_length%num_microbatches))
#make X_train and X_test divisible by num_microvatckes
X_train = X_train[:training_length]
y_train = y_train[:training_length]
X_test = X_test[:testing_length]
y_test = y_test[:testing_length]





model = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(num_classes, activation='softmax')])
loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
if dpsgd:
    optimizer = tensorflow_privacy.DPKerasAdamOptimizer(l2_norm_clip=l2_norm_clip,num_microbatches=num_microbatches,noise_multiplier=noise_multiplier,learning_rate=learning_rate)
else :
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
print("Train on {training_length} samples, validate on {testing_length} samples".format(training_length=training_length, testing_length=testing_length))
history = model.fit(X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        batch_size=batch_size)
train_loss_history = history.history['loss']
val_loss_history = history.history['val_loss']
train_accuracy_history = history.history['accuracy']
val_accuracy_history = history.history['val_accuracy']
loss, accuracy = model.evaluate(X_test, y_test)

privacy_report=compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(number_of_examples=training_length,
                                            batch_size=batch_size,
                                            noise_multiplier=0.8,
                                            num_epochs=epochs,
                                            delta=1e-5,
                                            used_microbatching=False)
eps,_ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(n=training_length, batch_size=batch_size, noise_multiplier=noise_multiplier, epochs=epochs, delta=delta)
print("The training is eps-delta private with (ε = {epsilon}, δ = {delta})".format(epsilon=round(eps,3), delta="{:e}".format(delta)))
metrics_df = {
    'dp-sdg': dpsgd,
    'batch_size': batch_size,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'epsilon': eps if dpsgd else 'NaN',
    'delta': "{:e}".format(delta) if dpsgd else 'NaN',
    'noise_multiplier': noise_multiplier if dpsgd else 'NaN',
    'num_microbatches': num_microbatches if dpsgd else 'NaN',
    'l2_norm_clip': l2_norm_clip if dpsgd else 'NaN',
    'train_loss': train_loss_history,
    'val_loss': val_loss_history,
    'train_accuracy':train_accuracy_history ,
    'val_accuracy': val_accuracy_history,
    'test_loss':loss,
    'test_accuracy':accuracy
}
existing_file_path = '../results/test_LargeNoise.csv'
existing_df = pd.read_csv(existing_file_path)
new_df = pd.DataFrame(metrics_df)
updated_df = pd.concat([existing_df, new_df], ignore_index=True)
updated_df.to_csv(existing_file_path, index=False)

epochs = range(1, epochs+1)
plt.plot(epochs, train_loss_history, 'g', label='Training loss')
plt.plot(epochs, val_loss_history, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../results/loss_LargeNoise.png')
plt.clf()
plt.plot(epochs, train_accuracy_history, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy_history, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../results/accuracy_LargeNoise.png')
plt.clf()
