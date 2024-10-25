import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

# Hyperparameters
NUM_WORDS = 5000
MAX_SEQUENCE_LEN = 500
EMBEDDING_DIM = 300  # Set to 300 to match GoogleNews-vectors
RNN_UNITS = 128
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20
K_FOLDS = 5


def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data = data[data['Rating'] != 3]
    plot_class_distribution(data)
    reviews = data['Review'].values
    ratings = data['Rating'].values
    reviews = [str(review) if pd.notna(review) else "" for review in reviews]
    labels = np.where(ratings <= 2, 0, 1)
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LEN)
    return padded_sequences, labels, tokenizer, reviews


def plot_class_distribution(data):
    class_counts = data['Rating'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()


def load_pretrained_word2vec(filepath):
    """
    Load pretrained Word2Vec embeddings.

    Args:
        filepath (str): Path to the pretrained Word2Vec binary file.

    Returns:
        KeyedVectors: Loaded Word2Vec model.
    """
    word2vec_model = KeyedVectors.load_word2vec_format(filepath, binary=True)
    return word2vec_model


def create_embedding_matrix(word_index, pretrained_word2vec):
    """
    Creates an embedding matrix using pretrained Word2Vec embeddings.

    Args:
        word_index (dict): Dictionary mapping words to their integer index.
        pretrained_word2vec (KeyedVectors): Pretrained Word2Vec model.

    Returns:
        np.array: Embedding matrix.
    """
    vocab_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i < NUM_WORDS:
            if word in pretrained_word2vec:
                embedding_matrix[i] = pretrained_word2vec[word]

    return embedding_matrix


def create_rnn_model(vocab_size, embedding_matrix):
    """
    Creates and compiles a Sequential SimpleRNN model with frozen Word2Vec embeddings.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_matrix (np.array): Pretrained Word2Vec embedding weights.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential()
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, trainable=False)
    model.add(embedding_layer)
    embedding_layer.build((None,))  # Manually build the layer before setting weights
    embedding_layer.set_weights([embedding_matrix])  # Set the embedding matrix as weights

    model.add(SimpleRNN(RNN_UNITS, return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def perform_kfold_cross_validation(X_train, y_train, tokenizer, embedding_matrix):
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []
    vocab_size = embedding_matrix.shape[0]
    class_weights = compute_class_weights(y_train)

    for train_index, val_index in kf.split(X_train):
        print(f'\nTraining fold {fold_no}...')
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = create_rnn_model(vocab_size, embedding_matrix)
        history = model.fit(X_train_fold, y_train_fold,
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_val_fold, y_val_fold),
                            class_weight=class_weights)

        plot_training_history(history, fold_no)

        val_preds = np.round(model.predict(X_val_fold)).flatten()
        accuracy = accuracy_score(y_val_fold, val_preds)
        print(f'Fold {fold_no} Accuracy: {accuracy * 100:.2f}%')
        accuracies.append(accuracy)
        fold_no += 1

    print(f'\nAverage Validation Accuracy: {np.mean(accuracies) * 100:.2f}%')
    return accuracies


def compute_class_weights(y_train):
    class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return {i: class_weights_array[i] for i in range(len(class_weights_array))}


def plot_training_history(history, fold_no):
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(f'Training and Validation Accuracy (Fold {fold_no})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def run_rnn_model(data_filepath, word2vec_filepath):
    padded_sequences, labels, tokenizer, reviews = load_and_prepare_data(data_filepath)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Load pretrained Word2Vec model
    pretrained_word2vec = load_pretrained_word2vec(word2vec_filepath)
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, pretrained_word2vec)

    # Perform K-fold cross-validation
    accuracies = perform_kfold_cross_validation(X_train, y_train, tokenizer, embedding_matrix)

    # Train and evaluate final model on the test set
    vocab_size = embedding_matrix.shape[0]
    final_model = create_rnn_model(vocab_size, embedding_matrix)
    final_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=compute_class_weights(y_train))

    test_preds = np.round(final_model.predict(X_test)).flatten()
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    return test_accuracy


# Run the model with paths to data and pretrained Word2Vec model
run_rnn_model("../data/reviews/clean_review_data.csv",
              "../data/word2vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin")
# C:\Users\olson\Documents\sentiment_analysis_repo\sentiment_analysis\sentiment_analysis\data\word2vec\GoogleNews-vectors-negative300.bin
