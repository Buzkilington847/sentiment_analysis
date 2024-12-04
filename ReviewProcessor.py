"""
Author: Andrew Olson

Purpose:
    This script processes a DataFrame containing product or service reviews with two columns:
    - 'Rating' (float): The numerical rating for the review.
    - 'Review' (string): The text of the review.

    It performs the following tasks:
    - Removes rows with missing ratings or empty reviews.
    - Removes stop words, emojis, and non-alphabetical characters from the review text.
    - Fixes smushed words
    - Assigns a label to each review: 1 if the rating is greater than 3, otherwise 0.
    - Outputs the processed reviews (lowercase) to a new CSV file (Rating, Label, Review).

    Additional Features:
    - Logs removed rows due to missing values or empty cleaned reviews.
    - Logs smushed or unrecognized words into "smushed.csv".
    - Checks words against a custom lexicon loaded from a CSV file.
"""

import pandas as pd
from nltk.corpus import stopwords
import emoji
import logging
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')


class ReviewProcessor:
    """
    A class to process reviews from a DataFrame.
    This class removes stop words, emojis, and non-alphabetical characters from reviews.
    It also assigns a label based on the rating and outputs the processed data to a new CSV file.
    """

    def __init__(self, data, lexicon):
        """
        Initialize the ReviewProcessor class with a DataFrame and a lexicon CSV file.

        Args:
            data (pd.DataFrame): A DataFrame containing 'Rating' and 'Review' columns.
            lexicon (str): Path to the CSV file containing valid English words.
        """
        self.data = data
        self.stop_words = set(stopwords.words('english'))
        self.lexicon = lexicon  # Load lexicon into a set
        self.smushed_words = set()  # Set to store unique smushed words

        # Configure logging
        logging.basicConfig(
            filename="data/logs/review_processor.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def check_word_in_lexicon(self, word):
        """
        Check if a word is in the custom lexicon.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word is in the lexicon, False otherwise.
        """
        if word.lower() not in self.lexicon:
            logging.info(f"Unrecognized word: {word}")
            print(f"Unrecognized word: {word}")
            self.smushed_words.add(word)  # Add to set of smushed words
            return False
        return True

    def remove_stopwords_and_emojis(self, review):
        """
        Remove stop words and emojis from a review.
        """
        review = emoji.replace_emoji(review, replace="")
        words = review.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def clean_review(self, review):
        """
        Clean and process the review text:
        - Remove non-alphabetical characters.
        - Check for unrecognized words.
        """
        processed_review = []

        for word in review.split():
            cleaned_word = ''.join([char for char in word if char.isalpha()])
            if cleaned_word:
                self.check_word_in_lexicon(cleaned_word)  # Log if not found in lexicon
                processed_review.append(cleaned_word)

        return ' '.join(processed_review)

    def process_reviews(self):
        """
        Process the DataFrame and output processed reviews, labels, and ratings.
        """
        logging.info(f"Initial number of rows: {len(self.data)}")

        # Drop rows with missing values
        before_cleaning = len(self.data)
        self.data = self.data.dropna(subset=['Rating', 'Review'])
        self.data = self.data[self.data['Review'].str.strip() != ""]
        logging.info(f"Removed {before_cleaning - len(self.data)} rows due to missing values.")

        # Initialize lists
        ratings, labels, reviews = [], [], []

        for index, row in self.data.iterrows():
            rating = row['Rating']
            review = row['Review']
            label = 1 if rating > 3 else 0

            review = self.remove_stopwords_and_emojis(review)
            review = self.clean_review(review)

            if not review:
                logging.warning(f"Row {index} removed: Review became empty after cleaning.")
                continue

            ratings.append(rating)
            labels.append(label)
            reviews.append(review.lower())

        # Create DataFrame for processed reviews
        processed_df = pd.DataFrame({
            "Rating": ratings,
            "Label": labels,
            "Review": reviews
        })
        processed_df.to_csv("data/reviews/processed_reviews.csv", index=False)

        # Save smushed words to CSV
        smushed_df = pd.DataFrame({"Unrecognized Word": list(self.smushed_words)})
        smushed_df.to_csv("data/reviews/smushed.csv", index=False)

        logging.info(f"Final number of rows in processed file: {len(processed_df)}")
        logging.info(f"Total smushed words logged: {len(self.smushed_words)}")

        # Return the processed DataFrame
        return processed_df
