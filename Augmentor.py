import pandas as pd
import random
from gensim.models import KeyedVectors
from deep_translator import GoogleTranslator

# Load the word vectors
loaded_word_vectors = KeyedVectors.load("data/fine_tuned_word_vectors.kv")


class Augmentor:
    def __init__(self, chunk_size=5, back_translation_language="fr"):
        self.chunk_size = chunk_size
        self.back_translation_language = back_translation_language

    def chunk_text(self, text):
        words = text.split()
        chunks = [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        return chunks

    def synonym_replacement(self, text):
        words = text.split()
        augmented_text = []
        for word in words:
            if word in loaded_word_vectors.key_to_index:
                similar_words = loaded_word_vectors.most_similar(word, topn=5)
                augmented_text.append(random.choice([w[0] for w in similar_words]))
            else:
                augmented_text.append(word)
        return " ".join(augmented_text)

    def random_deletion(self, text, probability=0.2):
        words = text.split()
        retained_words = [word for word in words if random.random() > probability]
        return " ".join(retained_words) if retained_words else text

    def back_translation(self, text):
        try:
            # Translate to foreign language
            translated = GoogleTranslator(source='en', target=self.back_translation_language).translate(text)
            # Translate back to English
            back_translated = GoogleTranslator(source=self.back_translation_language, target='en').translate(translated)
            return back_translated
        except Exception as e:
            print(f"Back translation failed for: '{text}'. Error: {e}")
            return text  # Return the original text if back translation fails

    def augment(self, text, methods):
        augmented_versions = []
        for method in methods:
            if method == "chunking":
                augmented_versions.extend(self.chunk_text(text))
            elif method == "synonym_replacement":
                augmented_versions.append(self.synonym_replacement(text))
            elif method == "random_deletion":
                augmented_versions.append(self.random_deletion(text))
            elif method == "back_translation":
                augmented_versions.append(self.back_translation(text))
        return augmented_versions


def process_reviews(csv_file, output_file, chunk_size=5, back_translation_language="fr", methods=None):
    if methods is None:
        methods = ["chunking", "synonym_replacement", "random_deletion", "back_translation"]

    # Load the CSV
    df = pd.read_csv(csv_file)

    # Extract and clean reviews
    reviews = [str(review) if isinstance(review, (str, float)) else "" for review in df.iloc[:, 1]]

    # Initialize the Augmentor
    augmentor = Augmentor(chunk_size=chunk_size, back_translation_language=back_translation_language)

    # Create a list to hold augmented data
    processed_data = []

    # Augment each review
    for review in reviews:
        augmented_versions = augmentor.augment(review, methods)  # Get augmented versions
        for augmented_review in augmented_versions:
            # Append original and augmented reviews for comparison
            processed_data.append({"Original Review": review, "Augmented Review": augmented_review})

    # Convert to DataFrame and save
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    process_reviews(
        csv_file="data/reviews/clean_review_data.csv",
        output_file="processed_reviews.csv",
        chunk_size=3,
        back_translation_language="es",
        methods=["chunking"]
    )

    # , "synonym_replacement", "random_deletion", "back_translation"
