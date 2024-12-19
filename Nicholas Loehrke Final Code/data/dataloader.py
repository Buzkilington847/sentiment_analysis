import torch
from torch.utils.data import Dataset, DataLoader
from config.config import config
import pandas as pd


class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        assert len(self.embeddings) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class HiddenStatesDataset(Dataset):
    def __init__(self, hidden_states, labels):
        self.hidden_states = hidden_states
        self.labels = labels
        assert len(self.hidden_states) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.labels[idx]


def create_original_sentiment_dataloader(df: pd.DataFrame, indices=None) -> DataLoader:
    if indices is None:
        fold_df = df
    else:
        fold_df = df.iloc[indices]
    
    df = df[df["augmented"] == False]

    fold_df.loc[:, "embedding"] = fold_df["embedding"].apply(lambda x: torch.tensor(x))
    features = list(fold_df["embedding"])
    labels = torch.tensor(fold_df["sentiment"].values)

    dataset = SentimentDataset(features, labels)
    return DataLoader(dataset, batch_size=config.rnn.batch_size, shuffle=True, pin_memory=True)

def create_augmented_sentiment_dataloader(df: pd.DataFrame, indices=None) -> DataLoader:
    if indices is None:
        fold_df = df
    else:
        fold_df = df.iloc[indices]
        
    # df = df[df["augmented"] == True]

    fold_df.loc[:, "embedding"] = fold_df["embedding"].apply(lambda x: torch.tensor(x))
    features = list(fold_df["embedding"])
    labels = torch.tensor(fold_df["sentiment"].values)

    dataset = SentimentDataset(features, labels)
    return DataLoader(dataset, batch_size=config.rnn.batch_size, shuffle=True, pin_memory=True)

def create_original_hidden_states_dataloader(df: pd.DataFrame, indices=None) -> DataLoader:
    if indices is None:
        fold_df = df
    else:
        fold_df = df.iloc[indices]
        
    df = df[df["augmented"] == False]

    fold_df.loc[:, "hidden_states"] = fold_df["hidden_states"].apply(lambda x: torch.tensor(x))
    features = list(fold_df["hidden_states"])
    labels = torch.tensor(fold_df["sentiment"].values)

    dataset = HiddenStatesDataset(features, labels)
    return DataLoader(dataset, batch_size=config.ffnn.batch_size, shuffle=True, pin_memory=True)

def create_augmented_hidden_states_dataloader(df: pd.DataFrame, indices=None) -> DataLoader:
    if indices is None:
        fold_df = df
    else:
        fold_df = df.iloc[indices]
        
    # df = df[df["augmented"] == True]

    fold_df.loc[:, "hidden_states"] = fold_df["hidden_states"].apply(lambda x: torch.tensor(x))
    features = list(fold_df["hidden_states"])
    labels = torch.tensor(fold_df["sentiment"].values)

    dataset = HiddenStatesDataset(features, labels)
    return DataLoader(dataset, batch_size=config.ffnn.batch_size, shuffle=True, pin_memory=True)