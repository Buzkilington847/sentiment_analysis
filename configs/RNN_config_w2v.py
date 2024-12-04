config = {
    "max_vocab_size": 15000,
    "max_sequence_length": 100,
    "embedding_dim": 300,
    "rnn_units": 64,
    "hidden_activation": "tanh",
    "output_activation": "sigmoid",
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "epochs": 10,
    "batch_size": 32,
    "k_folds": 5,
    "embedding_matrix": "data/word2vec/google_news/GoogleNews-vectors-negative300.bin"
}

# config = {
#     "max_vocab_size": 10000,
#     "max_sequence_length": 200,  # Determined from the dataset
#     "embedding_dim": 300,
#     "rnn_units": 64,
#     "hidden_activation": "tanh",
#     "output_activation": "softmax",  # For multi-class classification
#     "optimizer": "adam",
#     "loss": "categorical_crossentropy",
#     "epochs": 10,
#     "batch_size": 32,
#     "k_folds": 5,
#     "embedding_matrix": "data/word2vec/google_news/GoogleNews-vectors-negative300.bin"
# }