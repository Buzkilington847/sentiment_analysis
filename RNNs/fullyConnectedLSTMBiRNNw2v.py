import os
import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Masking, Bidirectional, Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from gensim.models import KeyedVectors
from RNNs.BaseRNNClassifier import BaseRNNClassifier


class FullyConnectedLSTMBiRNNw2vClassifier(BaseRNNClassifier):
    """
    Fully Connected RNN with two Bidirectional LSTM stages, using pre-trained Word2Vec embeddings.
    """

    def __init__(self, config):
        """
        Initialize the FullyConnectedLSTMBiRNNw2vClassifier.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config)  # Initialize the parent class
        self.config = config      # Store the configuration dictionary
        self.embedding_matrix = None  # Initialize the embedding matrix

    def build_model(self):
        """
        Build the architecture for the Fully Connected Bidirectional LSTM model.
        """
        vocab_size, embedding_dim = self.embedding_matrix.shape

        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,))

        # Embedding layer
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[self.embedding_matrix],
            trainable=False
        )(input_layer)

        # Masking layer to ignore padding (`^`)
        masked_embedding = Masking(mask_value=0.0)(embedding_layer)

        # First Bidirectional LSTM
        rnn1_output = Bidirectional(
            LSTM(units=self.rnn_units, activation="tanh", return_sequences=True)
        )(masked_embedding)

        # Second Bidirectional LSTM (fully connected)
        rnn2_output = Bidirectional(
            LSTM(units=self.rnn_units, activation="tanh", return_sequences=False)
        )(rnn1_output)

        # Output layer (binary classification)
        output_layer = Dense(units=1, activation="sigmoid")(rnn2_output)

        # Compile the model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss=self._custom_loss,
            metrics=["accuracy"]
        )
        print("Fully Connected LSTM BiRNN w2v model built and compiled.")

    def preprocess_data(self, reviews, labels):
        """
        Tokenize, pad, and truncate the reviews. Map labels to binary format.
        Args:
            reviews (list of str): The reviews to preprocess.
            labels (list of int): The corresponding labels.

        Returns:
            tuple: Tokenized and padded review data, and processed labels.
        """
        max_review_length = min(
            self.config.get("max_sequence_length", 100),
            max(len(review.split()) for review in reviews)
        )
        print(f"Dynamic maximum review length determined: {max_review_length}")

        # Replace padding character `^` with a unique token
        reviews = [
            " ".join(review.split()[:max_review_length]).ljust(max_review_length, "^")
            if len(review.split()) < max_review_length
            else " ".join(review.split()[:max_review_length])
            for review in reviews
        ]

        # Tokenize reviews
        self.tokenizer.fit_on_texts(reviews)
        tokenized_reviews = self.tokenizer.texts_to_sequences(reviews)

        # Padding and truncation
        padded_sequences = pad_sequences(
            tokenized_reviews, maxlen=max_review_length, padding="post", truncating="post"
        )

        # Replace `^` with a zero token (ignored in masking)
        vocab_size = len(self.tokenizer.word_index) + 1
        self.tokenizer.word_index["^"] = 0  # Add `^` as a padding token
        self.embedding_matrix = self._initialize_embedding_matrix(vocab_size)

        # Return raw labels and padded sequences
        return np.array(padded_sequences), np.array(labels)

    @staticmethod
    def _custom_loss(y_true, y_pred):
        """
        Custom loss to ignore blank labels.
        """
        mask = tf.cast(y_true != -1, tf.float32)  # Create a mask for non-blank labels
        bce = BinaryCrossentropy()(y_true * mask, y_pred * mask)  # Compute loss only for valid labels
        return tf.reduce_mean(bce)

    def _initialize_embedding_matrix(self, vocab_size):
        """
        Initialize the embedding matrix using Word2Vec.

        Args:
            vocab_size (int): Vocabulary size.

        Returns:
            np.ndarray: The initialized embedding matrix.
        """
        word2vec_path = self.config["embedding_matrix"]
        print(f"Loading Word2Vec model from {word2vec_path}...")
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        embedding_dim = self.config["embedding_dim"]
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        for word, index in self.tokenizer.word_index.items():
            if word in word2vec:
                embedding_matrix[index] = word2vec[word]
            else:
                embedding_matrix[index] = np.random.normal(size=(embedding_dim,))
        print(f"Embedding matrix created with shape {embedding_matrix.shape}")
        return embedding_matrix
