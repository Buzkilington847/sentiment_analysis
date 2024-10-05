import csv
import re
from nltk import ngrams


# Function to tokenize text into words without numbers and special characters
def tokenize(text):
    """
    Tokenize text into words without numbers and special characters.

    Parameters:
    text (str): The text to be tokenized.

    Returns:
    list: A list of words.
    """
    # Use regular expression to find words without numbers and special characters
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


# Function to generate all possible bigrams from a list of words
def generate_bigrams(words):
    """
    Generate all possible bigrams from a list of words.

    Parameters:
    words (list): The list of words.

    Returns:
    list: A list of bigrams.
    """
    return [' '.join(bigram) for bigram in ngrams(words, 2)]


# Initialize an empty set to store unique bigrams
bigram_set = set()


def clean_string(text):
    """
    Clean a string by converting to lowercase and removing numbers and punctuation.

    Parameters:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    # Lowercase the string
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


# Read the CSV file containing the reviews
with open('review_data.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    for row in csv_reader:
        review_text = row[1]  # Assuming the review text is in the second column
        words = tokenize(review_text)
        bigrams = generate_bigrams(words)

        bigram_set.update(bigram for bigram in bigrams if bigram in review_text)

print(len(bigram_set))

# Save the unique bigrams as comma-separated values in a single line
with open('clean_bigrams.csv', 'w', newline='', encoding='utf-8') as file:
    file.write(','.join(sorted(bigram_set)))
print("Unique bigrams saved to clean_bigrams.csv")
