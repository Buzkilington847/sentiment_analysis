# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import regularizers
# from sklearn.metrics import accuracy_score
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import GridSearchCV
# from keras.callbacks import EarlyStopping
# from sklearn.metrics import confusion_matrix
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.metrics import confusion_matrix
# import csv_tasks
# import classifier
#
# # Discarded imports
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from kerastuner.tuners import RandomSearch
#
# print(tf.config.list_physical_devices('GPU'))
# # Load the MNIST dataset
# df = pd.read_csv("../data/DNN/all_freq_test_unigram.csv")
# # df = pd.read_csv("squashed_unigram_mask.csv")
# y = pd.read_csv("../data/DNN/actual_labels_unigram.csv")
#
#
# def plot_loss(history):
#     """
#     Plots the training and validation loss versus epochs.
#
#     Args:
#     - history: History object returned by model.fit(), containing training/validation loss values.
#
#     Returns:
#     - None
#     """
#
#     # Extract training and validation loss from the history object
#     training_loss = history.history['loss']
#     validation_loss = history.history['val_loss']
#
#     # Create an array of epoch numbers
#     epochs = range(1, len(training_loss) + 1)
#
#     # Plot loss vs epochs
#     plt.plot(epochs, training_loss, 'b', label='Training Loss')
#     plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
#
#
# def compute_accuracy(actual_labels, predicted_labels):
#     """
#     Compute the accuracy of predicted labels compared to actual labels.
#
#     Args:
#     actual_labels (list): List of actual labels.
#     predicted_labels (list): List of predicted labels.
#
#     Returns:
#     float: Accuracy of the predicted labels.
#     """
#     # Convert lists to numpy arrays if they're not already
#     actual_labels = np.array(actual_labels)
#     predicted_labels = np.array(predicted_labels)
#
#     # Compute accuracy
#     accuracy = accuracy_score(actual_labels, predicted_labels)
#
#     return accuracy
#
#
# def convert_labels_to_probabilities(labels):
#     """
#     Convert a 1D array of binary labels into a two-column array of probabilities.
#
#     Args:
#     labels (numpy.ndarray): 1D array of binary labels (0 for negative, 1 for positive).
#
#     Returns:
#     numpy.ndarray: Two-column array of probabilities, where the first column represents the probability of being positive
#                    and the second column represents the probability of being negative.
#     """
#     # Initialize the probabilities array
#     if isinstance(labels, pd.DataFrame):
#         labels = labels.values
#
#     labels = labels.reshape(-1, 1)
#     probabilities = np.zeros((len(labels), 2))
#
#     # Set the first column to represent the probability of being positive
#     probabilities[:, 0] = 1 - labels.ravel()  # ravel() is used to convert the array to 1D
#
#     # Set the second column to represent the probability of being negative
#     probabilities[:, 1] = labels.ravel()
#
#     return probabilities
#
#
# # Define a function to create the model
# def build_model(hp):
#     model = keras.Sequential()
#     model.add(layers.Dense(units=hp.Int('units_layer1', min_value=1, max_value=20, step=2),
#                            activation='relu', input_shape=(X_train.shape[1],)))
#     # model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
#     model.add(layers.Dense(units=hp.Int('units_layer2', min_value=0, max_value=20, step=2),
#                            activation='relu'))
#     # model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
#     model.add(layers.Dense(2, activation='softmax'))
#
#     model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
#
# def run_neural_network():
#     df = df.transpose()
#     X = df.values  # Input features (frequencies)
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#
#     # Further split the training data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
#     # test= convert_labels_to_probabilities(y_test)
#     train = convert_labels_to_probabilities(y_train)
#     y_val = convert_labels_to_probabilities(y_val)
#     # Define the model
#     model = models.Sequential([
#         layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
#         layers.Dense(12, activation='relu'),
#         layers.Dense(8, activation='relu'),
#         layers.Dense(4, activation='relu'),
#         layers.Dense(2, activation='softmax')
#     ])
#     # opt = tf.keras.optimizers.Adam(learning_rate=0.0004)
#     # Compile the model
#     model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
#
#     # Define your batch size
#     batch_size = 32
#     early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
#     # Train the model on GPU
#     with tf.device('/GPU:0'):
#         history = model.fit(X_train, train, epochs=50, batch_size=batch_size, validation_data=(X_val, y_val),
#                             callbacks=[early_stopping])
#
#     predictions = model.predict(X_test)
#     save_to_csv(predictions, "prediction.csv")
#     predicted_labels = [1 if pred[1] > pred[0] else 0 for pred in predictions]
#     save_to_csv(y_test, "labels_actual.csv")
#     acc = compute_accuracy(y_test, predicted_labels)
#     print(acc)
#     # Evaluate the model on the test set ,validation_data=(X_val, y_val)
#     cm = confusion_matrix(y_test, predicted_labels)
#     cm = rotate_2x2(cm)
#     print(cm)
#     plot_loss(history)
#
#     df = df.transpose()
#     X = df.values  # Input features (frequencies)
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#
#     # Further split the training data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
#     # test= convert_labels_to_probabilities(y_test)
#     train = convert_labels_to_probabilities(y_train)
#     y_val = convert_labels_to_probabilities(y_val)
#     # Define the model
#     model = models.Sequential([
#         layers.Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
#         # layers.BatchNormalization(),
#
#         layers.Dense(2, activation='relu'),
#         # layers.BatchNormalization(),
#         # layers.Dense(4, activation='relu'),
#         # layers.Dense(8, activation='relu'),
#
#         layers.Dense(2, activation='softmax')
#     ])
#     opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#     # Compile the model
#     model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
#
#     # Define your batch size
#     batch_size = 32
#     early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
#     # Train the model on GPU
#     with tf.device('/GPU:0'):
#         history = model.fit(X_train, train, epochs=65, batch_size=batch_size, validation_data=(X_val, y_val),
#                             callbacks=[early_stopping])
#
#     predictions = model.predict(X_test)
#     save_to_csv(predictions, "prediction.csv")
#     predicted_labels = [1 if pred[1] > pred[0] else 0 for pred in predictions]
#     save_to_csv(predicted_labels, "labels_predict.csv")
#     acc = compute_accuracy(y_test, predicted_labels)
#     print(acc)
#     # Evaluate the model on the test set ,validation_data=(X_val, y_val)
#     cm = confusion_matrix(y_test, predicted_labels)
#     cm = rotate_2x2(cm)
#     print(cm)
#     plot_loss(history)
#
#     df = df.transpose()
#     X = df.values  # Input features (frequencies)
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#
#     # Further split the training data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
#     # test= convert_labels_to_probabilities(y_test)
#     train = convert_labels_to_probabilities(y_train)
#     y_val = convert_labels_to_probabilities(y_val)
#     # Initialize the RandomSearch tuner
#     tuner = RandomSearch(
#         build_model,
#         objective='val_accuracy',
#         max_trials=20,  # Number of hyperparameter combinations to try
#         executions_per_trial=1,
#         directory='my_dir',  # Directory to save the tuning logs and checkpoints
#         project_name='my_project'
#     )
#
#     # Perform hyperparameter tuning
#     tuner.search(X_train, train,
#                  epochs=45,
#                  validation_data=(X_val, y_val),
#                  callbacks=[
#                      keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)])
#
#     # Get the best hyperparameters
#     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#     print(f"Best Hyperparameters: {best_hps}")
#
#     # Retrieve the best model
#     best_model = tuner.get_best_models(num_models=1)[0]
#     best_model.summary()
#
#
# run_neural_network()
