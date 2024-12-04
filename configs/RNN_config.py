"""
Hyperparameters Configuration:

max_vocab_size (int):
    Maximum number of unique words in the vocabulary.
    - Larger values allow the model to consider more unique words, which can improve performance on diverse datasets.
    - Smaller values reduce memory usage and computational cost.

max_sequence_length (int):
    Maximum length of a sequence (number of words in a review) after padding or truncation.
    - Longer sequences retain more information but may lead to overfitting or increased computational cost.
    - Shorter sequences may truncate useful context but improve training speed.

embedding_dim (int):
    Size of the embedding vector for each word.
    - Larger dimensions capture more complex word relationships but require more computational resources and data.
    - Smaller dimensions make the model faster but may lose critical word semantics.

rnn_units (int):
    Number of neurons in the SimpleRNN layer.
    - More units allow the model to capture more intricate patterns in sequential data, improving performance.
    - Fewer units make the model simpler and faster to train but might underfit complex datasets.

hidden_activation (str):
    Activation function for the hidden layer in the RNN.
    - Common choices are:
        - "relu" (default): Efficient and helps avoid vanishing gradients, suitable for most tasks.
        - "tanh": Useful for sequential data with positive and negative values but can slow down training.

output_activation (str):
    Activation function for the output layer.
    - "sigmoid" (default): Outputs a probability between 0 and 1, suitable for binary classification.
    - Use "softmax" for multi-class classification.

optimizer (str):
    Optimizer for training.
    - "adam" (default): Works well in most scenarios, combining the benefits of other optimizers.
    - Alternatives include "sgd" for simplicity or "rmsprop" for sequential data like text.

loss (str):
    Loss function to quantify prediction errors.
    - "binary_crossentropy" (default): Suitable for binary classification tasks.
    - Use "categorical_crossentropy" for multi-class classification.

epochs (int):
    Number of complete passes through the dataset during training.
    - Higher values allow the model to learn more patterns but may lead to overfitting.
    - Fewer epochs risk underfitting, where the model doesn’t learn enough from the data.

batch_size (int):
    Number of samples per gradient update.
    - Smaller sizes improve generalization but slow down training.
    - Larger sizes make training faster but may require more memory and risk overfitting.

k_folds (int):
    Number of folds for K-Fold Cross-Validation.
    - Higher values provide more robust evaluation but increase computational time.
    - Lower values reduce computational cost but might make evaluation less reliable.
"""

config = {
    "max_vocab_size": 7000,
    "max_sequence_length": 500,
    "embedding_dim": 300,
    "rnn_units": 64,
    "hidden_activation": "relu",
    "output_activation": "sigmoid",
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "epochs": 10,
    "batch_size": 32,
    "k_folds": 5
}
