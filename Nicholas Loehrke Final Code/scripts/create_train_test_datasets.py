import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def augment_reviews(args):
    df = pd.read_csv(args.input_file)
    df = df.dropna()
    
    train_df, test_df = train_test_split(df, test_size=args.test_size, stratify=df["rating"])
    
    train_df.to_csv(args.train_outfile, index=False)
    test_df.to_csv(args.test_outfile, index=False)


if __name__ == "__main__":
    print(f"++++++++++++++{__file__}++++++++++++++")
    
    parser = argparse.ArgumentParser(description="Split dataset into stratified samples.")

    # File arguments
    parser.add_argument("input_file", help="Input file to split.")
    parser.add_argument("--train_outfile", default="raw_training_dataset.csv", help="Training output file")
    parser.add_argument("--test_outfile", default="raw_testing_dataset.csv", help="Testing output file")
    parser.add_argument("-s", "--test_size", type=float, default=0.2, help="Test split percentage")
    args = parser.parse_args()
        
    augment_reviews(args)