from dataclasses import dataclass
from pathlib import Path


@dataclass
class RNN:
    # hyperparameters
    input_size: int = 300
    hidden_size: int = 64
    num_layers: int = 2
    bias: bool = False
    dropout: float = 0.1
    bidirectional: bool = False

    # training parameters
    epochs: int = 50
    folds: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 1


@dataclass
class FFNN:
    # hyperparameters
    hidden_state_embedding_size: int = 400
    input_size: int = RNN.hidden_size * hidden_state_embedding_size

    # training parameters
    epochs: int = 20
    folds: int = 5
    learning_rate: float = 0.001
    batch_size: int = 256


@dataclass
class Paths:
    fasttext: str = str(Path("fasttext/crawl-300d-2M-subword.bin").resolve())


@dataclass
class Config:
    rnn: RNN
    ffnn: FFNN
    paths: Paths


config = Config(rnn=RNN(), paths=Paths(), ffnn=FFNN())


# # For fast testing

# from dataclasses import dataclass
# from pathlib import Path


# @dataclass
# class RNN:
#     # hyperparameters
#     input_size: int = 300
#     hidden_size: int = 16
#     num_layers: int = 1
#     bias: bool = False
#     dropout: float = 0.1
#     bidirectional: bool = False

#     # training parameters
#     epochs: int = 1
#     folds: int = 2
#     learning_rate: float = 0.0001
#     batch_size: int = 1


# @dataclass
# class FFNN:
#     # hyperparameters
#     hidden_state_embedding_size: int = 10
#     input_size: int = RNN.hidden_size * hidden_state_embedding_size

#     # training parameters
#     epochs: int = 1
#     folds: int = 2
#     learning_rate: float = 0.001
#     batch_size: int = 256


# @dataclass
# class Paths:
#     fasttext: str = str(Path("fasttext/crawl-300d-2M-subword.bin").resolve())


# @dataclass
# class Config:
#     rnn: RNN
#     ffnn: FFNN
#     paths: Paths


# config = Config(rnn=RNN(), paths=Paths(), ffnn=FFNN())

