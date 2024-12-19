import pandas as pd
import argparse
import random


def chunk(review, size, max_chunks=None):
    words = review.split()
    chunks = [words[i : i + size] for i in range(0, len(words), size)]

    if max_chunks:
        chunks = chunks[:max_chunks]

    return [" ".join(chunk) for chunk in chunks]


def shuffle_review(review, size):
    chunks = chunk(review, size)
    # random.shuffle(chunks)
    for i in range(0, len(chunks) - 1, 2):
        chunks[i], chunks[i + 1] = chunks[i + 1], chunks[i]
    return " ".join(chunks)


def augment_reviews(args):
    df = pd.read_csv(args.input_file)
    df = df.dropna()

    new_reviews = []
    new_ratings = []
    new_augmented_flags = []

    for _, row in df.iterrows():
        old_review = row["review"]
        old_rating = row["rating"]

        # --no_original
        if not args.no_original:
            new_reviews.append(old_review)
            new_ratings.append(old_rating)
            new_augmented_flags.append(False)

        # --max_chunks
        max_chunks = None
        if args.max_chunks:
            max_chunks = args.max_chunks[0]

        # --chunk
        if args.chunk is not None:
            new_chunks = chunk(old_review, args.chunk[0], max_chunks)
            for new_review in new_chunks:
                new_reviews.append(new_review)
                new_ratings.append(old_rating)
                new_augmented_flags.append(True)

                
        # --shuffle_chunk
        if args.shuffle_chunk is not None:
            new_review = shuffle_review(old_review, args.shuffle_chunk[0])
            new_reviews.append(new_review)
            new_ratings.append(old_rating)
            new_augmented_flags.append(True)
                
    # Create a new DataFrame with augmented data
    augmented_df = pd.DataFrame({"rating": new_ratings, "review": new_reviews, "augmented": new_augmented_flags})

    # Save the augmented data to a new CSV file
    augmented_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    print(f"++++++++++++++{__file__}++++++++++++++")
    
    parser = argparse.ArgumentParser(description="Text Augmentation Script for Preprocessing")

    # File arguments
    parser.add_argument("input_file", help="Input file for text processing.")
    parser.add_argument("-o", "--output_file", default="preprocessed_reviews.csv", help="Optional output file.")

    # Augmentation options
    parser.add_argument("--no_original", action="store_true", help="Exclude original reviews from output.")
    parser.add_argument("--chunk", type=int, nargs=1, metavar="<size>", help="Chunk the text into fixed size chunks.")
    parser.add_argument(
        "--shuffle_chunk", type=int, nargs=1, metavar="<size>", help="Chunk the text and shuffle the chunks."
    )
    parser.add_argument(
        "--max_chunks", type=int, nargs=1, metavar="<max_chunks>", help="Max number of chunks generated per review."
    )

    args = parser.parse_args()

    if args.max_chunks and not args.chunk:
        parser.error("--argument --max_chunks: requires --chunk to be specified")
        
    augment_reviews(args)