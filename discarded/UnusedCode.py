"""
def generate_review_dictionary(dictionary, reviews):
    review_dict = []
    for review in reviews:
        words = review.split()
        for word in words:
            word_cleaned = word.lower().strip(string.punctuation)
            if word_cleaned in dictionary and word_cleaned not in review_dict:
                review_dict.append(word_cleaned)
    return sorted(review_dict)




"""