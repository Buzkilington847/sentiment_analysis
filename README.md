# borilml
Project overview: Classify Amazon product reviews as either positive or negative based on the text used
within. The pipeline started with a web-scraper which retrieved review text and star rating on the online 
retailer site, Amazon. We were able to scrape 7,102 reviews ranging from 1-5 stars (excluding 
3 stars as they were deemed neutral and not helpful toward positive and negative classification).
Once the reviews were successfully scraped and stored into a csv file with two columns: Rating and Review. 
The Rating column contains the number of stars (stored as an integer rounded to the nearest whole number) 
that a review received. Further in the pipeline, the dataset was split into training and test sets. 
All of the reviews were used to generate two dictionaries: uni-grams and bi-grams. These dictionaries contain
only unique, legal words within the English lexicon.  Using the dictionary's size, two equal-sized bags of words
(will be referenced as BoW from here on) were created. These are 1 dimensional (1 row to many columns) and each
cell corresponds to the word found at the same index in the dictionary. All cells in both BoW have a default value of "0". 
These values are then incremented if an occurrence of the word associated with the given cell is found.
Once finished, the BoW are then normalized to create frequencies of word usage using the formula Frequency = (wordFreq / totalWordsInDictionary). 
The normalized frequencies are then further used to develop a mask and filter, drastically improving results. 
The mask takes the absolute value of the difference between a word's frequency in both the negative and positive
BoW ([pFreq - nFreq]) and divides the result by the summation of two frequencies. 
This mask is then applied to both BoW, thus effectively reducing the impact that certain words (namely 
those that might appear in both positive and negative reviews equally as frequently like articles)
both positive and negative reviews. The filter is designed off the mask since it can be applied to
filter out words that were used roughly the same number of times in both positive and
negative reviews. Ideally this will filter out words that fall under categories such as: conjunctions, pronouns, and
prepositions, articles, etc... After this filter is applied, the testing data that was set aside is used. The
testing data will then take each individual review, compare it to the dictionary designed
earlier to create its own bag of words. The bag of words will then be compared to both the
negative and positive bag of words designed from the testing set. The comparison is done
by using cosine distancing and then the review will be classified based on which cosine
distance is smaller. This process is done for each individual review. The original star
reviews are still stored within the system, so that is compared to the classifier to determine
if it was correct or not. Our system then will output its overall unweighted accuracy, EER, F1
score, confusion matrix, and precision. Using the EER, an optimal threshold is applied to
increase the accuracy of the system.
