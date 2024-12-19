import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import pandas as pd

from models.ffnn import FFNN
from data.dataloader import create_original_hidden_states_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    ffnn = FFNN()

    state_dict = torch.load(model_path, map_location=device)
    ffnn.load_state_dict(state_dict)
    ffnn.to(device)
    ffnn.eval()
    return ffnn

def evaluate(model: nn.Module, dataloader: DataLoader):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for embedding, label in dataloader:
            embedding, label = embedding.to(device), label.to(device)

            outputs = model(embedding.float())

            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    metric = {}
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    TN, FP, FN, TP = confusion_matrix(all_labels, all_outputs).ravel()

    metric = {
        "unweighted_accuracy": (TP + TN) / (TP + TN + FP + FN),
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "mcc": ((TP * TN) - (FP * FN)) / float(np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),
    }

    return metric

def main(args):
    ffnn = load_model(args.model_file)
    df = pd.read_pickle(args.embeddings_file)
    dataloader = create_original_hidden_states_dataloader(df)
    
    metric = evaluate(ffnn, dataloader)
    
    print(f'Unweighted accuracy: {metric["unweighted_accuracy"]:.3f}')
    print(f'TN: {metric["TN"]}')
    print(f'FP: {metric["FP"]}')
    print(f'FN: {metric["FN"]}')
    print(f'TP: {metric["TP"]}')
    print(f'mcc: {metric["mcc"]:.3f}')


if __name__ == "__main__":
    print(f"++++++++++++++{__file__}++++++++++++++")
    
    parser = argparse.ArgumentParser(description="Evalaute FFNN")
    parser.add_argument("model_file", type=str, help="Input model file")
    parser.add_argument("embeddings_file", type=str, help="Input embeddings file in pickle format")

    args = parser.parse_args()

    main(args)
