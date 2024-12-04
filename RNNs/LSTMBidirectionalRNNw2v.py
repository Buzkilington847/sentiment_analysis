from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from gensim.models import KeyedVectors
import numpy as np
from RNNs.BaseRNNClassifier import BaseRNNClassifier


class LSTMBidirectionalRNNw2vClassifier(BaseRNNClassifier):
    """
    Bidirectional LSTM Classifier with pre-trained Word2Vec embeddings.
    """

    def __init__(self, config):
        """
        Initialize the LSTMBidirectionalRNNw2vClassifier.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config)
        self.config = config
        self.embedding_matrix = None  # Initialize the embedding matrix as None

    def build_model(self):
        """
        Build the architecture for the Bidirectional LSTM model with Word2Vec embeddings.
        """
        vocab_size, embedding_dim = self.embedding_matrix.shape
        self.model = Sequential([
            # Embedding layer initialized with pre-trained embeddings
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=self.max_sequence_length,
                weights=[self.embedding_matrix],
                trainable=False  # Keep the embeddings fixed
            ),
            # Bidirectional LSTM layer
            Bidirectional(
                LSTM(units=self.rnn_units, activation=self.hidden_activation, return_sequences=False)
            ),
            # Dense output layer
            Dense(
                units=1,  # Assuming binary classification
                activation=self.output_activation
            )
        ])
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
        print("Bidirectional LSTM w2v model built and compiled.")

    def preprocess_data(self, reviews, labels):
        """
        Tokenize and pad the reviews. Load and initialize the embedding matrix.

        Args:
            reviews (list of str): The reviews to preprocess.
            labels (list or np.ndarray): The corresponding labels.

        Returns:
            tuple: Tokenized and padded review data, and the labels.
        """
        # Tokenize and pad reviews
        padded_sequences, labels = super().preprocess_data(reviews, labels)

        # Load Word2Vec embeddings
        word2vec_path = self.config["embedding_matrix"]
        print(f"Loading Word2Vec model from {word2vec_path}...")
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        # Initialize the embedding matrix
        vocab_size = len(self.tokenizer.word_index) + 1
        embedding_dim = self.config["embedding_dim"]
        self.embedding_matrix = np.zeros((vocab_size, embedding_dim))

        for word, index in self.tokenizer.word_index.items():
            if word in word2vec:
                self.embedding_matrix[index] = word2vec[word]
            else:
                self.embedding_matrix[index] = np.random.normal(size=(embedding_dim,))
        print(f"Embedding matrix created with shape {self.embedding_matrix.shape}")

        return padded_sequences, labels
