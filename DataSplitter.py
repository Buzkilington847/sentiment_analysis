from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np


class DataSplitter:
    """
    A utility class for splitting data into training, validation, and test sets
    using K-Fold Cross-Validation with optional stratification.
    """
    def __init__(self, n_splits=5, test_size=0.2, random_state=42, shuffle=True, stratified=False):
        """
        Initialize the DataSplitter.

        Args:
            n_splits (int): Number of folds for K-Fold Cross-Validation.
            test_size (float): Fraction of data to reserve for the test set.
            random_state (int): Seed for reproducibility.
            shuffle (bool): Whether to shuffle data before splitting.
            stratified (bool): Whether to use stratified splitting.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratified = stratified

    def split(self, data, labels):
        """
        Split data into training and validation sets using K-Fold.

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
        if self.stratified:
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        for train_idx, val_idx in kf.split(data, labels):
            yield data[train_idx], data[val_idx], labels[train_idx], labels[val_idx]

    def train_test_split(self, data, labels):
        """
        Split data into training and test sets.

        Args:
            data (np.ndarray): Input features.
            labels (np.ndarray): Corresponding labels.

        Returns:
            tuple: Train and test data and labels.
        """
        if self.stratified:
            return train_test_split(data, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels)
        else:
            return train_test_split(data, labels, test_size=self.test_size, random_state=self.random_state, shuffle=self.shuffle)
