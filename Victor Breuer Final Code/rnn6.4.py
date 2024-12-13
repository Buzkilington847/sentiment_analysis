














import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
import gensim
from scipy.stats import mode
from tensorflow.keras.layers import Input, TimeDistributed, Flatten

# Step 1: Load the CSV file and filter out 3-star (neutral) reviews
data_path = 'clean_review_data.csv'  # Updated file path
df = pd.read_csv(data_path)

# Remove neutral reviews (3 stars)
df = df[df['Rating'] != 1]

# Step 2: Split data into sentences and categorical labels
corpus = df['Review'].tolist()
labels = df['Rating'].apply(lambda x: 1 if x == 2 else 0).values  # Map 2 to 1 (positive), and 0 stays 0 (negative)

# Convert labels to categorical for multi-class classification
labels = tf.keras.utils.to_categorical(labels, num_classes=2)

# Step 3: Train a word2vec model on the corpus
word2vec_model = gensim.models.Word2Vec(sentences=[sentence.split() for sentence in corpus], vector_size=100, window=5, min_count=1, workers=4)

# Step 4: Prepare tokenizer and sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(corpus)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Step 5: Create an embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Step 6: Build the RNN model with return_sequences=False
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(SimpleRNN(units=128, return_sequences=False))  # Output only at the last time step
model.add(Dense(units=2, activation='softmax'))  # Two output neurons for multi-class classification

# Compile the model for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Create TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
dataset = dataset.shuffle(buffer_size=len(padded_sequences)).batch(2)  # Adjust batch size as needed

# Step 8: Split the dataset into training, validation, and testing sets
train_size = int(0.60 * len(padded_sequences))  # 60% for training
val_size = int(0.20 * len(padded_sequences))    # 20% for validation
test_size = len(padded_sequences) - train_size - val_size  # Remaining 20% for testing

# Split padded_sequences and labels
train_sequences = padded_sequences[:train_size]
train_labels = labels[:train_size]
val_sequences = padded_sequences[train_size:train_size + val_size]
val_labels = labels[train_size:train_size + val_size]
test_sequences = padded_sequences[train_size + val_size:]
test_labels = labels[train_size + val_size:]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels)).batch(2)
val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels)).batch(2)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(2)

# Debugging: Check dataset sizes
print(f"Training dataset size: {len(list(train_dataset))}")
print(f"Validation dataset size: {len(list(val_dataset))}")
print(f"Testing dataset size: {len(list(test_dataset))}")

# Step 9: Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Step 10: Train the model with early stopping
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, 
                    callbacks=[early_stopping], verbose=1)

# Step 11: Extract the outputs from the RNN model
# Create a new model that outputs the RNN layer's outputs
rnn_output_model = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)

# Use the trained model to get RNN outputs
train_rnn_outputs = rnn_output_model.predict(train_sequences)
val_rnn_outputs = rnn_output_model.predict(val_sequences)
test_rnn_outputs = rnn_output_model.predict(test_sequences)

# Step 12: Define the secondary neural network
# Input shape is based on the RNN outputs
dense_units = 128  # Example value
num_classes = 2
secondary_input = Input(shape=(train_rnn_outputs.shape[1],))  # Shape is 2D: (batch_size, units)
secondary_dense_layer = Dense(units=dense_units, activation='relu')(secondary_input)
secondary_output = Dense(units=num_classes, activation='softmax')(secondary_dense_layer)

secondary_model = tf.keras.Model(inputs=secondary_input, outputs=secondary_output)

# Step 13: Compile the secondary model
secondary_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 14: Train the secondary model
history_secondary = secondary_model.fit(
    train_rnn_outputs, train_labels,
    epochs=50,
    validation_data=(val_rnn_outputs, val_labels),
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Step 15: Evaluate the secondary model on the test set
test_accuracy_secondary = secondary_model.evaluate(test_rnn_outputs, test_labels, batch_size=32, verbose=1)[1]
print(f"Test Accuracy (Secondary Model): {test_accuracy_secondary:.4f}")
