import string  # Import the 'string' module to handle string manipulation, like removing punctuation
import pandas as pd  # Import pandas to handle CSV operations and data structures
from num2words import num2words  # Import num2words to convert numbers to words (e.g., '2' to 'two')
import emoji  # Import emoji module to handle and replace emoji characters
import re  # Import regular expressions for advanced filtering, such as removing numbers and non-ASCII characters


def save_to_csv(array, file_name):
    """
    Save the given array to a CSV file as a single row with many columns.

    Parameters:
        array (list or numpy array): The array to be saved as a single row.
        file_name (str): The name of the CSV file (include .csv extension).

    Returns:
        None
    """
    # Convert the array into a DataFrame with a single row (each n-gram as a column)
    df = pd.DataFrame([array])
    # Save the DataFrame without headers or index to get only n-grams in the file
    df.to_csv(file_name, index=False, header=False)


def generate_ngrams(text, ngram_size):
    """
    Generate n-grams of size ngram_size from the input text.

    Parameters:
    text (str): The input review as a string.
    ngram_size (int): The n-gram size (e.g., 1 for unigram, 2 for bigram, etc.)

    Returns:
    list: A list of n-grams generated from the text.
    """
    # Split the text into words
    words = text.split()
    ngrams = []

    # Return an empty list if the text is too short to form any n-grams
    if len(words) < ngram_size:
        return []

    # Generate n-grams by taking consecutive words of size 'ngram_size'
    for i in range(len(words) - ngram_size + 1):
        ngram = ' '.join(words[i:i + ngram_size])  # Join 'ngram_size' words to form an n-gram
        ngrams.append(ngram)

    return ngrams


def clean_review(review):
    """
    Clean the review text by removing punctuation, replacing emojis with spaces,
    removing numbers, non-ASCII characters, and handling possible compound words.

    Parameters:
    review (str): The input review as a string.

    Returns:
    str: Cleaned review string.
    """
    # Replace emojis in the review with spaces
    review_no_emojis = emoji.replace_emoji(review, replace=' ')

    # Convert the review to lowercase and remove punctuation
    review_no_punctuation = review_no_emojis.lower().translate(str.maketrans('', '', string.punctuation))

    # Remove any numbers from the review
    review_no_numbers = re.sub(r'\d+', '', review_no_punctuation)

    # Remove non-ASCII characters to ensure text is clean and consistent
    review_ascii_only = re.sub(r'[^\x00-\x7F]+', '', review_no_numbers)

    return review_ascii_only


def generate_dictionary(ngram_size, reviews):
    """
    Generate a dictionary of n-grams from a list of reviews.

    Parameters:
    ngram_size (int): Size of the n-gram (e.g., 1 for unigram, 2 for bigram, etc.)
    reviews (list): List of reviews.

    Returns:
    list: Sorted list of unique n-grams.
    """
    ngram_set = set()  # Use a set to automatically handle duplicate n-grams

    for review in reviews:
        if isinstance(review, str):  # Ensure the review is a valid string
            # Clean the review by removing punctuation, emojis, numbers, and non-ASCII characters
            review_cleaned = clean_review(review)
            # Generate n-grams from the cleaned review
            ngrams = generate_ngrams(review_cleaned, ngram_size)
            # Add the generated n-grams to the set (avoids duplicates)
            ngram_set.update(ngrams)

    # Convert the set of unique n-grams into a sorted list
    sorted_ngrams = sorted(ngram_set)

    # Save the sorted n-grams as a CSV file (one row, many columns)
    save_to_csv(sorted_ngrams, f"../data/dictionaries/ngram_dictionary_{num2words(ngram_size)}.csv")

    return sorted_ngrams


def load_review_data(path):
    """
    Load review data from a CSV file.

    Parameters:
    path (str): Path to the CSV file.

    Returns:
    list: List of reviews as strings.
    """
    # Load the review data from the CSV file into a DataFrame
    reviews_df = pd.read_csv(path)

    # Assuming reviews are in the second column of the CSV, convert that column to a list of strings
    reviews = reviews_df.iloc[:, 1].tolist()

    # Ensure that each review is a valid string and filter out any NaN values
    reviews = [str(review) for review in reviews if pd.notna(review)]

    return reviews


def create_dictionary(gram_size):
    """
    A main function that guides the user to create an n-gram dictionary by loading review data
    and generating a dictionary based on user-specified n-gram size.
    """

    # Load the review data from the CSV file
    reviews = load_review_data("../data/reviews/clean_review_data.csv")

    # Generate a dictionary of n-grams from the reviews
    generate_dictionary(int(gram_size), reviews)


# Uncomment the following line to run the function when the script is executed
# create_dictionary(5)
