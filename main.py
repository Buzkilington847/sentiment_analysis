"""
Author: Andrew Olson

Purpose:
    This script serves as the entry point to process reviews, train sentiment analysis models,
    and generate relevant metrics and graphs for evaluation.

    - Interacts with accessory files and modules for data preprocessing, training, and evaluation.
    - Delegates model-specific tasks to the respective model classes.

Output:
    - A trained RNN model saved to disk.
    - Metrics and visualizations for analysis.

Notes:
    dictionary.csv data taken from https://github.com/dwyl/english-words/blob/master/words.txt
"""
import os
import pandas as pd
import pickle
import numpy as np

from DataSplitter import DataSplitter
from MetricAnalysis import MetricAnalysis
from RNNs.RNNFactory import RNNFactory
from configs.RNN_config import config as RNN_config
from configs.RNN_config_w2v import config as RNN_config_w2v


def load_lexicon(lexicon_path):
    """
    Load the lexicon file into a set for word filtering.

    Args:
        lexicon_path (str): Path to the lexicon CSV file.

    Returns:
        set: A set of valid words from the lexicon.
    """
    if not os.path.exists(lexicon_path):
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")
    return set(pd.read_csv(lexicon_path)["Word"].str.lower())


def analyze_metrics(output_metrics_dir, k_folds):
    """
    Analyze and visualize metrics for each fold.

    Args:
        output_metrics_dir (str): Directory to save metrics plots.
        k_folds (int): Number of K-Folds used in training.
    """
    for fold_no in range(1, k_folds + 1):
        history_file = os.path.join("history_logs", f"fold_{fold_no}_history.pkl")
        predictions_file = os.path.join("predictions", f"fold_{fold_no}_predictions.pkl")

        if os.path.exists(history_file) and os.path.exists(predictions_file):
            with open(history_file, "rb") as f:
                history = pickle.load(f)

            with open(predictions_file, "rb") as f:
                y_true, y_pred = pickle.load(f)

            # Generate and save plots and summaries
            MetricAnalysis.plot_metrics(history, output_dir=output_metrics_dir, prefix=f"fold_{fold_no}")
            MetricAnalysis.summarize_metrics(history, fold_no=fold_no)
            MetricAnalysis.plot_confusion_matrix(y_true, y_pred, output_dir=output_metrics_dir, prefix=f"fold_{fold_no}")
            MetricAnalysis.classification_report_summary(y_true, y_pred)


def main():
    input_path = "data/reviews/processed_reviews.csv"
    word2vec_path = "data/word2vec/google_news/GoogleNews-vectors-negative300.bin"
    output_metrics_dir = "models/bidirectional_rnn_w2v/"
    model_save_path = "models/bidirectional_rnn_w2v/model.h5"

    # Load and preprocess data
    data = pd.read_csv(input_path)
    print(f"Normalized Data columns: {data.columns}")

    reviews = data["Review"].values
    labels = data["Label"].values

    # Create the RNN model
    model = RNNFactory.create_rnn("fullyConnectedLSTMBiRNNw2v", RNN_config_w2v)
    preprocessed_data, preprocessed_labels = model.preprocess_data(reviews, labels)

    # Reserve a test set
    splitter = DataSplitter(n_splits=RNN_config_w2v["k_folds"], test_size=0.2, stratified=True)
    train_data, test_data, train_labels, test_labels = splitter.train_test_split(preprocessed_data, preprocessed_labels)

    # Train the model using K-Fold Cross-Validation
    print("Starting cross-validation...")
    model.train_model(train_data, train_labels, splitter)

    # Evaluate the model on the test set
    print("Evaluating on the test set...")
    test_loss, test_accuracy = model.model.evaluate(test_data, test_labels, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Generate predictions for the test set
    y_pred = (model.model.predict(test_data) > 0.5).astype("int32")

    # Use MetricAnalysis for detailed metrics
    print("Generating metrics and plots...")
    MetricAnalysis.plot_confusion_matrix(test_labels, y_pred, output_dir=output_metrics_dir, prefix="test_set")
    MetricAnalysis.classification_report_summary(test_labels, y_pred)

    # Save the trained model
    print(f"Saving the trained model to {model_save_path}...")
    model.save_model(model_save_path)


if __name__ == "__main__":
    main()
