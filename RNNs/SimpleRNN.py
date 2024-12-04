from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from RNNs.BaseRNNClassifier import BaseRNNClassifier


class SimpleRNNClassifier(BaseRNNClassifier):
    """
    A basic implementation of a single-directional RNN model.
    """

    def build_model(self):
        """
        Build the architecture for the Simple RNN model.
        """
        self.model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.max_vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            # SimpleRNN layer
            SimpleRNN(
                units=self.rnn_units,
                activation=self.hidden_activation
            ),
            # Dense output layer
            Dense(
                units=1,  # Assuming binary classification
                activation=self.output_activation
            )
        ])
        # Compile the model with optimizer and loss from the configuration
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
        print("Simple RNN model built and compiled.")
