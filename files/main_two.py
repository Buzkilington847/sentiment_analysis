"""
Authors: Andrew, Grant, Kyle
System: Sentiment Analysis Pipeline for Amazon Product Reviews
About:

TODO:
- Delete discarded files when ready
- Delete review_data.csv and only use "clean_review_data"
- Implement ngram generation into pipeline
- Reimplement BoW into pipeline
- Reimplement DNN into pipeline
- Implement RNN using word2vec
- Reimplement Wordcloud
- Implement Word network
- Implement graph generations for statistics
"""

import bag_of_words
import rnn_model


def main():
    continue_program = True

    while continue_program:
        mode = input("Options are:\n"
                     "1. 'B' for Bag of Words\n"
                     "2. 'D' for Deep Neural Network\n"
                     "3. 'R' for Recurrent Neural Network\n"
                     "Enter the classifier mode: ")

        if mode == "B":
            bag_of_words.start()
        elif mode == "D":
            pass
        elif mode == "R":
            # Call the RNN model function from rnn_model module
            data_filepath = "../data/reviews/clean_review_data.csv"
            rnn_model.run_rnn_model(data_filepath)
        elif mode == "E":
            break
        else:
            print("Invalid command!")
    return -1


if __name__ == "__main__":
    main()
