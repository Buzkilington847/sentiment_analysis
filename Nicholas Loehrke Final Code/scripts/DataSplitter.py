import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


class DataSplitter:
    """
    A utility class for splitting data into training and validation sets
    using K-Fold Cross-Validation with optional stratification.

    Attributes:
        n_splits (int): Number of folds for K-Fold Cross-Validation.
        random_state (int): Seed for reproducibility.
        shuffle (bool): Whether to shuffle data before splitting.
        stratified (bool): Whether to use stratified splitting.
    """

    def __init__(self, n_splits=5, random_state=42, shuffle=True, stratified=False):
        """
        Initialize the DataSplitter.

        Args:
            n_splits (int): Number of folds for K-Fold Cross-Validation.
            random_state (int): Seed for reproducibility.
            shuffle (bool): Whether to shuffle data before splitting.
            stratified (bool): Whether to use stratified splitting.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratified = stratified

        if self.stratified:
            self.splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            self.splitter = KFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )

    def split(self, data, labels):
        """
        Split data into training and validation sets.

        Args:
            data (np.ndarray): Input features.
            labels (np.ndarray): Corresponding labels.

        Yields:
            tuple: A tuple containing:
                - train_data (np.ndarray): Training feature set.
                - val_data (np.ndarray): Validation feature set.
                - train_labels (np.ndarray): Training labels.
                - val_labels (np.ndarray): Validation labels.
        """
        for train_idx, val_idx in self.splitter.split(data, labels):
            yield data[train_idx], data[val_idx], labels[train_idx], labels[val_idx]
