import torch
import numpy as np
import pandas as pd
import argparse

from models.rnn import RNN
from data.dataloader import create_augmented_sentiment_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    rnn = RNN()

    state_dict = torch.load(model_path, map_location=device)
    rnn.load_state_dict(state_dict)
    rnn.to(device)
    rnn.eval()
    return rnn

def main(args):
    rnn = load_model(args.model_file)
    df = pd.read_pickle(args.embeddings_file)
    dataloader = create_augmented_sentiment_dataloader(df)
    all_hidden_states = []
    sentiments = []

    with torch.no_grad():
        for embedding, sentiment in dataloader:
            embedding = embedding.to(device).float()

            _, hidden_states = rnn(embedding)
            hidden_states = hidden_states[0, :, :]

            embedding_length, hidden_size = hidden_states.shape

            embedding_length = min(embedding_length, args.dimensions)

            padded_hidden_states = torch.zeros((args.dimensions, hidden_size), device=device)

            padded_hidden_states[:embedding_length, :] = hidden_states[:embedding_length, :]

            flattened_hidden_state = padded_hidden_states.flatten().cpu().numpy()
            all_hidden_states.append(flattened_hidden_state)
            sentiments.extend(sentiment.cpu().numpy())
            
            
    out_df = pd.DataFrame({
        "hidden_states": all_hidden_states,
        "sentiment": sentiments,
        "augmented": df["augmented"]
    })

    out_df.to_pickle(args.output)

    print(f'Hidden state array dimensions: {len(all_hidden_states[0])}')

if __name__ == "__main__":
    print(f"++++++++++++++{__file__}++++++++++++++")
    
    parser = argparse.ArgumentParser(description="Extract hidden states")
    parser.add_argument("model_file", type=str, help="Input model file")
    parser.add_argument("embeddings_file", type=str, help="Input embeddings file")
    parser.add_argument("-d", "--dimensions", type=int, default=1000, help="Maximum length for padding embeddings.")
    parser.add_argument("-o", "--output", type=str, default="hidden_states.pkl", help="Output file")

    args = parser.parse_args()

    main(args)
