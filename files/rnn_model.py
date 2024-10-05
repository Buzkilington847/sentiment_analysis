import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def run_rnn_model(data_filepath):
    # Load data
    data = pd.read_csv(data_filepath)

    # Remove rows with neutral (3) ratings
    data = data[data['Rating'] != 3]

    # Plot class distribution
    class_counts = data['Rating'].value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()

    # Extract reviews and ratings
    reviews = data['Review'].values
    ratings = data['Rating'].values

    # Clean and prepare the reviews (handle missing values)
    reviews = [str(review) if pd.notna(review) else "" for review in reviews]

    # Binarize the ratings (1, 2 -> 0, 4, 5 -> 1)
    labels = np.where(ratings <= 2, 0, 1)

    # Tokenize the text data
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)

    # Pad sequences to ensure uniform length
    max_sequence_len = 300
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_sequence_len)

    # Print tokenized sample
    print(f"Sample tokenized sequences: {sequences[:5]}")
    print(f"Sample padded sequences: {padded_sequences[:5]}")
    print(f"Sample labels: {labels[:5]}")

    # Split the dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    print(f"Total number of samples in training set: {len(X_train)}")
    print(f"Total number of samples in test set: {len(X_test)}")

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=[review.split() for review in reviews], vector_size=64, window=5, min_count=1, workers=4)
    word2vec_weights = word2vec_model.wv.vectors

    # Ensure the tokenizer's word index matches the Word2Vec model's vocabulary
    vocab_size = min(len(tokenizer.word_index) + 1, word2vec_weights.shape[0])

    # Compute class weights to handle class imbalance
    class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}

    def create_rnn_model():
        model = Sequential()
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=word2vec_weights.shape[1], trainable=False)
        model.add(embedding_layer)
        embedding_layer.build((None,))  # Build the layer to initialize weights
        embedding_layer.set_weights([word2vec_weights[:vocab_size]])
        model.add(Bidirectional(LSTM(64, return_sequences=False)))  # Using a Bidirectional LSTM
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))  # Added an additional Dense layer with 64 units
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification (positive/negative)

        # Using Adam optimizer with a learning rate of 0.001
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Perform k-fold cross-validation (k=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []

    for train_index, val_index in kf.split(X_train):
        print(f'\nTraining fold {fold_no}...')
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Print the number of samples being processed in each fold
        print(f"Number of samples in training fold {fold_no}: {len(X_train_fold)}")
        print(f"Number of samples in validation fold {fold_no}: {len(X_val_fold)}")

        # Create and train the RNN model
        model = create_rnn_model()
        history = model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=64, validation_data=(X_val_fold, y_val_fold), class_weight=class_weights)

        # Plot training and validation accuracy per epoch
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title(f'Training and Validation Accuracy (Fold {fold_no})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Evaluate the model on the validation data
        val_preds = model.predict(X_val_fold)
        val_preds = np.round(val_preds).flatten()  # Convert probabilities to binary 0/1
        accuracy = accuracy_score(y_val_fold, val_preds)
        print(f'Fold {fold_no} Accuracy: {accuracy * 100:.2f}%')
        accuracies.append(accuracy)
        fold_no += 1

    # Average accuracy across all folds
    print(f'\nAverage Validation Accuracy: {np.mean(accuracies) * 100:.2f}%')

    # Train final model on the entire training set and evaluate on test data
    final_model = create_rnn_model()
    final_model.fit(X_train, y_train, epochs=10, batch_size=64, class_weight=class_weights)

    # Evaluate on the test set
    test_preds = final_model.predict(X_test)
    test_preds = np.round(test_preds).flatten()
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    return test_accuracy
