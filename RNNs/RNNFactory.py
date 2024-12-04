# Importing the additional RNN classes
from RNNs.LSTMBidirectionalRNNw2v import LSTMBidirectionalRNNw2vClassifier
from RNNs.SimpleRNN import SimpleRNNClassifier
from RNNs.SimpleRNNw2v import SimpleRNNw2vClassifier
# from RNNs.BidirectionalRNN import BidirectionalRNNClassifier
from RNNs.BidirectionalRNNw2v import BidirectionalRNNw2vClassifier
from RNNs.fullyConnectedLSTMBiRNNw2v import FullyConnectedLSTMBiRNNw2vClassifier


# from RNNs.LSTMBidirectionalRNN import LSTMBidirectionalRNNClassifier
# from RNNs.LSTMBidirectionalRNNw2v import LSTMBidirectionalRNNw2vClassifier
# from RNNs.FullyConnectedLSTMBiRNNw import FullyConnectedLSTMBiRNNwClassifier
# from RNNs.FullyConnectedLSTMBiRNNw2v import FullyConnectedLSTMBiRNNw2vClassifier
# from RNNs.EnsembleLSTM import EnsembleLSTMClassifier
# from RNNs.EnsembleLSTMw2v import EnsembleLSTMw2vClassifier


class RNNFactory:
    """
    Factory class for creating RNN models with a consistent interface.
    """

    @staticmethod
    def create_rnn(model_type, config):
        """
        Create an RNN model based on the specified type.

        Args:
            model_type (str): Type of the RNN model.
                              Options: "SimpleRNN", "SimpleRNNw2v",
                                       "BidirectionalRNN", "BidirectionalRNNw2v",
                                       "LSTMBidirectionalRNN", "LSTMBidirectionalRNNw2v",
                                       "fullyConnectedLSTMBiRNNw", "fullyConnectedLSTMBiRNNw2v",
                                       "ensembleLSTM", "ensembleLSTMw2v".
            config (dict): Configuration dictionary for the model.

        Returns:
            BaseRNNClassifier: An instance of the specified RNN model.

        Raises:
            ValueError: If the model_type is not supported.
        """
        if model_type == "SimpleRNN":
            return SimpleRNNClassifier(config)
        elif model_type == "SimpleRNNw2v":
            return SimpleRNNw2vClassifier(config)
        # elif model_type == "BidirectionalRNN":
        #     return BidirectionalRNNClassifier(config)
        elif model_type == "BidirectionalRNNw2v":
            return BidirectionalRNNw2vClassifier(config)
        # elif model_type == "LSTMBidirectionalRNN":
        #     return LSTMBidirectionalRNNClassifier(config)
        elif model_type == "LSTMBidirectionalRNNw2v":
            return LSTMBidirectionalRNNw2vClassifier(config)
        # elif model_type == "fullyConnectedLSTMBiRNN":
        #     return FullyConnectedLSTMBiRNNwClassifier(config)
        elif model_type == "fullyConnectedLSTMBiRNNw2v":
            return FullyConnectedLSTMBiRNNw2vClassifier(config)
        # elif model_type == "ensembleLSTM":
        #     return EnsembleLSTMClassifier(config)
        # elif model_type == "ensembleLSTMw2v":
        #     return EnsembleLSTMw2vClassifier(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
