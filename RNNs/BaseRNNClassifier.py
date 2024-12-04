import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class BaseRNNClassifier(ABC):
    """
    Abstract Base Class for RNN models to enforce consistency in methods.
    """

    def __init__(self, config):
        self.max_vocab_size = config["max_vocab_size"]
        self.max_sequence_length = config["max_sequence_length"]
        self.embedding_dim = config["embedding_dim"]
        self.rnn_units = config["rnn_units"]
        self.hidden_activation = config["hidden_activation"]
        self.output_activation = config["output_activation"]
        self.optimizer = config["optimizer"]
        self.loss = config["loss"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.k_folds = config["k_folds"]

        self.model = None
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<OOV>")

    @abstractmethod
    def build_model(self):
        """
        Build the RNN model architecture. Must be implemented by subclasses.
        """
        pass

    def preprocess_data(self, reviews, labels):
        """
        Tokenize and pad the reviews.

        Args:
            reviews (list of str): The reviews to preprocess.
            labels (list or np.ndarray): The corresponding labels.

        Returns:
            tuple: Tokenized and padded review data, and the labels.
        """
        self.tokenizer.fit_on_texts(reviews)
        sequences = self.tokenizer.texts_to_sequences(reviews)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding="post")
        return np.array(padded_sequences), np.array(labels)

    def train_model(self, data, labels, splitter):
        """
        Train the model using K-Fold Cross-Validation.

        Args:
            data (np.ndarray): Input features.
            labels (np.ndarray): Corresponding labels.
            splitter (DataSplitter): A DataSplitter instance for K-Fold splitting.
        """
        if self.model is None:
            self.build_model()

        os.makedirs("history_logs", exist_ok=True)
        os.makedirs("predictions", exist_ok=True)

        fold_no = 1
        for train_data, val_data, train_labels, val_labels in splitter.split(data, labels):
            print(f"\nTraining Fold {fold_no}/{self.k_folds}...")

            # Train the model
            history = self.model.fit(
                train_data, train_labels,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(val_data, val_labels),
                verbose=1
            )

            # Save history and predictions
            history_file = os.path.join("history_logs", f"fold_{fold_no}_history.pkl")
            predictions_file = os.path.join("predictions", f"fold_{fold_no}_predictions.pkl")

            with open(history_file, 'wb') as hist_file:
                pickle.dump(history.history, hist_file)

            predictions = self.model.predict(val_data)
            with open(predictions_file, 'wb') as pred_file:
                pickle.dump(predictions, pred_file)

            fold_no += 1

    def save_model(self, save_path):
        """
        Save the trained model to a file.

        Args:
            save_path (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained.")
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

        tokenizer_path = save_path + "_tokenizer.pkl"
        with open(tokenizer_path, 'wb') as file:
            pickle.dump(self.tokenizer, file)
        print(f"Tokenizer saved to {tokenizer_path}")

    def load_model(self, model_path, tokenizer_path):
        """
        Load a trained model and tokenizer.

        Args:
            model_path (str): Path to the saved model file.
            tokenizer_path (str): Path to the saved tokenizer file.
        """
        from tensorflow.keras.models import load_model

        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError("Model or tokenizer file not found.")

        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")

        with open(tokenizer_path, 'rb') as file:
            self.tokenizer = pickle.load(file)
        print(f"Tokenizer loaded from {tokenizer_path}")
