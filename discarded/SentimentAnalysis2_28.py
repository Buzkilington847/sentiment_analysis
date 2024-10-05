import csv
import string
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, det_curve, DetCurveDisplay, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from numpy.linalg import norm

import csv
import string
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, det_curve, DetCurveDisplay, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from numpy.linalg import norm
from tabulate import tabulate


def load_csv_column(csv_file, column_index, first_row_header=False):
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[:, column_index]


def load_csv_row(csv_file, row_index, first_row_header=False):
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[row_index]


def generate_review_dictionary(dictionary, reviews):
    review_dict = []
    for review in reviews:
        words = review.split()
        for word in words:
            word_cleaned = word.lower().strip(string.punctuation)
            if word_cleaned in dictionary and word_cleaned not in review_dict:
                review_dict.append(word_cleaned)
    return sorted(review_dict)


def generate_bag_of_words_frequencies(dictionary, reviews):
    dictionary_mapping = {word: index for index, word in enumerate(dictionary)}
    bag_words = np.zeros(len(dictionary))
    total_words = 0
    # print("type one")
    # print(type(reviews))
    # print("Reviews")
    # print(reviews.shape)
    for review_text in reviews:
        # print("Review_text.shape")
        # print(review_text[1])
        # print("type 2")
        # print(type(review_text))
        if isinstance(review_text, str) and review_text.lower() != 'nan':
            words = review_text.split()  # Split review text into words
            # print("Words in the current review:", words)
            cleaned_words = [word.lower().strip(string.punctuation) for word in words]
            # print("Cleaned words:", cleaned_words)
            for word in cleaned_words:
                index = dictionary_mapping.get(word)
                if index is not None:
                    total_words += 1
                    bag_words[index] += 1
    # print("Total words processed in the review:", total_words)
    # print("Words in the dictionary:", list(dictionary_mapping.keys()))
    return [(word_count / total_words) for word_count in bag_words] if total_words > 0 else np.zeros(len(dictionary))


def cosine_similarity_scores(all_frequencies):
    positive_column = all_frequencies[:, 0].reshape(1, -1)
    negative_column = all_frequencies[:, 1].reshape(1, -1)
    pos_similarities = []
    neg_similarities = []
    for i in range(2, all_frequencies.shape[1]):
        test_column = all_frequencies[:, i].reshape(1, -1)
        positive_similarity = cosine_similarity(test_column, positive_column)[0, 0]
        negative_similarity = cosine_similarity(test_column, negative_column)[0, 0]
        pos_similarities.append(positive_similarity)
        neg_similarities.append(negative_similarity)
    return pos_similarities, neg_similarities


def classify_labels(positive_scores, negative_scores, threshold):
    labels = np.where((positive_scores - negative_scores - threshold) <= 0, 0, 1)
    scores = positive_scores - negative_scores
    return labels, scores


def label_classifier(pscore, nscore, threshold):
    label = np.zeros(len(pscore), dtype=int)
    scores = np.zeros(len(pscore), dtype=float)
    # assigns labels as Positive(1) or Negative(0)
    for i in range(len(pscore)):
        prediction = pscore[i] - nscore[i] - threshold
        scores[i] = pscore[i] - nscore[i]
        if prediction <= 0:
            label[i] = 0
        else:
            label[i] = 1

    return label, scores


def calculate_performance_matrix(inputdata, flag):
    class_accuracy_matrix = np.empty(len(inputdata))
    for i in range(len(inputdata)):
        class_accuracy = (inputdata[i, i] / (np.sum(inputdata[i, :]))) * 100
        class_accuracy_matrix[i] = + class_accuracy

    Uar = round((1 / len(class_accuracy_matrix)) * (np.sum(class_accuracy_matrix)), 2)
    if flag == 1:
        print("Unweighted Accuracy:", Uar)

    # Proportion of true positive predictions in all positive predictions
    if np.sum(inputdata[:, 0]) == 0:
        Precision = 0
    else:
        Precision = round((inputdata[0, 0] / np.sum(inputdata[:, 0])), 2)
        if flag == 1:
            print("Precision:", Precision)
    # Proportion of true positive predictions made by the model out of all positive samples in the dataset
    if np.sum(inputdata[0, :]) == 0:
        Recall = 0
    else:
        Recall = round((inputdata[0, 0] / np.sum(inputdata[0, :])), 2)
        if flag == 1:
            print("Recall:", Recall)
    # Defined as the harmonic mean of precision and recall
    F1_Score = round((2 / ((1 / Precision) + (1 / Recall))), 2) if Precision != 0 and Recall != 0 else 0
    if flag == 1:
        print("F1 Score:", F1_Score, "\n")

    return Uar


def calculate_det(fpr, fnr):
    display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="DET Curve")
    display.plot()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.show()


def count_correct_predictions(actual_labels, predicted_labels):
    correct_count = 0
    wrong_count = 0
    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == predicted:
            correct_count += 1
        else:
            wrong_count += 1
    return correct_count, wrong_count


def confusion_matrix_scheduler(actual_labels, positive_scores, negative_scores, thresholds):
    uar_matrix = []
    confusion_matrices = []  # Initialize an empty list to store confusion matrices

    all_fpr = []
    all_fnr = []

    for i, threshold in enumerate(thresholds):
        # calculate predicted label and scores
        predicted_labels, calc_scores = label_classifier(positive_scores, negative_scores, threshold)
        # calculate the Confusion Matrix
        cm = metrics.confusion_matrix(actual_labels, predicted_labels)
        cm = rotate_2x2(cm)
        Uar = calculate_performance_matrix(cm, 0)
        fpr, fnr, _ = det_curve(actual_labels, calc_scores)

        confusion_matrices.append(cm)  # Append the confusion matrix
        uar_matrix.append(Uar)
        all_fpr.extend(fpr)  # Accumulate false positive rates
        all_fnr.extend(fnr)  # Accumulate false negative rates

    EER = calculate_EER(all_fpr, all_fnr)
    return predicted_labels, EER, thresholds, confusion_matrices, uar_matrix, all_fpr, all_fnr


def generate_freq(review, review_dict):
    neg_testing = review[review[:, 0] < 3]
    pos_testing = review[review[:, 0] > 3]

    pos_frequencies_test = generate_bag_of_words_frequencies(review_dict, pos_testing[:, 1])
    pos_frequencies_test = np.array(pos_frequencies_test)
    pos_frequencies_test = pos_frequencies_test.reshape(-1, 1)

    neg_frequencies_test = generate_bag_of_words_frequencies(review_dict, neg_testing[:, 1])
    neg_frequencies_test = np.array(neg_frequencies_test)
    neg_frequencies_test = neg_frequencies_test.reshape(-1, 1)
    return pos_frequencies_test, neg_frequencies_test


def find_max(list):
    max_value = max(list)
    max_index = list.index(max_value)

    return max_index


def plot_threshold_vs_accuracy(threshold_values, uar_values):
    # Plotting the threshold_matrix(x axis) vs unweighted accuracy (UAR)
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, uar_values, marker='o', linestyle='-')
    plt.title('Threshold vs Unweighted Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Unweighted Accuracy (UAR)')
    plt.grid(True)
    plt.show()


def statistics(actual_labels, predicted_labels):
    # Generate a classification report
    report = classification_report(actual_labels, predicted_labels, zero_division=1)
    return report


# Calculates the weight mask and returns it
def calculate_mask(negative_freq, positive_freq):
    delta = abs(positive_freq - negative_freq)
    summation = negative_freq + positive_freq
    modified_array = np.where(summation == 0, 1, summation)
    weights = delta / modified_array
    return weights


def calculate_filter(mask_temp, percent_filter):
    num_elements_to_zero = int(len(mask_temp) * percent_filter)
    # Sort the array and find the threshold value
    sorted = np.sort(mask_temp, axis=0)
    filter_threshold = sorted[num_elements_to_zero]
    # Set elements less than the threshold to zero
    new_zeros = np.where(mask_temp < filter_threshold)[0]
    mask_temp[mask_temp < filter_threshold] = 0
    return mask_temp, num_elements_to_zero


def calculate_EER(fpr, fnr):
    EER_Matrix = np.empty(len(fpr))
    for i in range(len(fpr)):
        eer = (fpr[i] + fnr[i]) / 2.0
        EER_Matrix[i] = eer
    best_eer = min(EER_Matrix)
    return best_eer


def calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold, lower_threshold, actual_labels):
    count = 0

    mid_threshold = (upper_threshold + lower_threshold) / 2
    high_predicted_label, high_calc_score = label_classifier(pos_cosine, neg_cosine, upper_threshold)
    low_predicted_label, low_calc_score = label_classifier(pos_cosine, neg_cosine, lower_threshold)
    mid_predicted_label, mid_calc_score = label_classifier(pos_cosine, neg_cosine, mid_threshold)

    cm_high = metrics.confusion_matrix(actual_labels, high_predicted_label)
    cm_low = metrics.confusion_matrix(actual_labels, low_predicted_label)
    cm_mid = metrics.confusion_matrix(actual_labels, mid_predicted_label)

    fpr_high, fnr_high = calculate_fpr_fnr(cm_high)
    fpr_low, fnr_low = calculate_fpr_fnr(cm_low)
    fpr_mid, fnr_mid = calculate_fpr_fnr(cm_mid)

    diff_mid = fpr_mid - fnr_mid

    if diff_mid > 0:
        lower_threshold = mid_threshold
    elif diff_mid < 0:
        upper_threshold = mid_threshold

    # print(lower_threshold, upper_threshold)
    while (abs(lower_threshold - upper_threshold) > 0.000002) or count == 1000:
        count = count + 1
        mid_threshold = (upper_threshold + lower_threshold) / 2
        high_predicted_label, high_calc_score = label_classifier(pos_cosine, neg_cosine, upper_threshold)
        low_predicted_label, low_calc_score = label_classifier(pos_cosine, neg_cosine, lower_threshold)
        mid_predicted_label, mid_calc_score = label_classifier(pos_cosine, neg_cosine, mid_threshold)

        cm_high = metrics.confusion_matrix(actual_labels, high_predicted_label)
        cm_low = metrics.confusion_matrix(actual_labels, low_predicted_label)
        cm_mid = metrics.confusion_matrix(actual_labels, mid_predicted_label)

        fpr_high, fnr_high = calculate_fpr_fnr(cm_high)
        fpr_low, fnr_low = calculate_fpr_fnr(cm_low)
        fpr_mid, fnr_mid = calculate_fpr_fnr(cm_mid)

        diff_mid = fpr_mid - fnr_mid

        if diff_mid > 0:
            lower_threshold = mid_threshold
        elif diff_mid < 0:
            upper_threshold = mid_threshold

        # print(lower_threshold, upper_threshold)
    mid_threshold = (upper_threshold + lower_threshold) / 2
    return mid_threshold


def binary_search(low, mid, high):
    if abs(low - mid) < abs(high - mid):
        return low, mid
    else:
        return mid, high


def create_actual_labels(ratings):
    actual_labels = []
    for rating in ratings:
        if rating > 3:
            actual_labels.append(1)
        else:
            actual_labels.append(0)
    return actual_labels


def calculate_fpr_fnr(confusion_matrix):
    # Extract values from confusion matrix
    TN, FP, FN, TP = confusion_matrix.ravel()
    # Calculate False Positive Rate (FPR)
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0
    # Calculate False Negative Rate (FNR)
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 0
    return FPR, FNR


def best_threshold(actual_labels, pos_cosine, neg_cosine, mid_threshold):
    predicted_labels, calc_scores = label_classifier(pos_cosine, neg_cosine, mid_threshold)
    # calculate the Confusion Matrix
    cm = metrics.confusion_matrix(actual_labels, predicted_labels)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])

    # cm_display.plot()
    # plt.show()
    FPR, FNR = calculate_fpr_fnr(cm)
    EER = (FPR + FNR) / 2

    true_acceptance = cm[1][1]  # True Acceptance (valid inputs accepted)
    false_rejection = cm[1][0]  # False Rejection (valid inputs rejected)
    false_acceptance = cm[0][1]  # False Acceptance (invalid inputs accepted)
    true_rejection = cm[0][0]  # True Rejection (invalid inputs rejected)

    print("The Threshold Used:", mid_threshold)
    print(f"Best Possible Confusion Matrix:\n{cm}")
    print(" Best Possible Equal Error Rate: ", EER)
    cm = rotate_2x2(cm)
    print(f"Rotated Possible Confusion Matrix:\n{cm}")

    total_valid_inputs = true_acceptance + false_rejection  # Total valid inputs
    total_invalid_inputs = false_acceptance + true_rejection  # Total invalid inputs

    far = false_acceptance / total_invalid_inputs
    frr = false_rejection / total_valid_inputs
    return far, frr


def rotate_2x2(matrix):
    # Swap elements diagonally
    rotated_matrix = np.array([[matrix[1][1], matrix[1][0]],
                               [matrix[0][1], matrix[0][0]]])

    return rotated_matrix


def load_data():
    reviews_matrix = pd.read_csv("review_data.csv").to_numpy().T
    review_dict = load_csv_row("review_dict.csv", 0, first_row_header=False)
    sorted_indices = np.argsort(reviews_matrix[0])
    reviews_matrix = reviews_matrix[:, sorted_indices]
    return reviews_matrix, review_dict


def main():
    # Hyperparameters
    percentage_testing = None
    percentage_training = None
    lower_threshold = None
    upper_threshold = None
    lower_bound = None
    upper_bound = None
    step_value = None

    # Other variables
    pos_frequencies_train = None
    neg_frequencies_train = None
    pos_frequencies_test = None
    neg_frequencies_test = None
    actual_labels = None
    predicted_labels = None
    EER = None
    threshold_values = None
    thresholds = None
    confusion_matrices = None
    report = None
    uar_matrix = None
    all_fpr = None
    all_fnr = None
    parameters_set = False

    # Load data
    reviews_matrix, review_dict = load_data()

    while True:
        if not parameters_set:
            print("Setting hyperparameters...")
            default = input("Enter Y for default settings): ").strip().upper()
            if default == "Y":
                percentage_testing = 0.1
                lower_threshold = -1
                upper_threshold = 1
                lower_bound = -0.1
                upper_bound = 0.1
                step_value = 0.02
            else:
                percentage_testing = float(input("Enter percentage for testing data (e.g., 0.1 for 10%): "))
                lower_threshold = float(input("Enter lower threshold value (Recommended value=-1): "))
                upper_threshold = float(input("Enter upper threshold value (Recommended value=1): "))
                lower_bound = float(input("Enter lower bound value (Recommended value=-0.1, an integer or float that's "
                                          "less than zero): "))
                upper_bound = float(input("Enter upper bound value (Recommended value=0.1, an integer or float that's "
                                          "greater than zero): "))
                step_value = float(input("Enter step value (Recommended value=0.02): "))
                threshold_values = np.arange(lower_bound, upper_bound, step_value)
            percentage_training = 1 - percentage_testing

            # Store hyperparameters in a list of tuples
            hyperparameters = [
                ("Percentage for testing data", percentage_testing),
                ("Percentage for training data", percentage_training),
                ("Lower threshold value", lower_threshold),
                ("Upper threshold value", upper_threshold),
                ("Lower bound value", lower_bound),
                ("Upper bound value", upper_bound),
                ("Step value", step_value)
            ]

            parameters_set = True

        print("Press 'H' for a list of commands.")
        command = input("Enter a command: ").strip().upper()

        if command == 'H':
            print("Commands:")
            print("M - Main function of the program")
            print("G - Graph generation")
            print("I - Input hyperparameters")
            print("P - Print values")
            print("E - Terminate program")
            print("F - Graph Filter")

        elif command == "F":
            print("Executing filter function...")

            if (percentage_testing is not None and lower_threshold is not None and upper_threshold is not None
                    and lower_bound is not None and upper_bound is not None and step_value is not None):

                # start_percent = input("Enter filter start percent in decimal form: ")
                # end_percent = input("Enter filter end percent in decimal form: ")
                # index_percent = input("Enter index amount in decimal form: ")

                sorted_indices = np.argsort(reviews_matrix[0])
                reviews_matrix = reviews_matrix[:, sorted_indices]

                review_training, review_testing = train_test_split(reviews_matrix.T, test_size=percentage_testing,
                                                                   random_state=42)
                neg_testing = review_testing[review_testing[:, 0] < 3]
                pos_testing = review_testing[review_testing[:, 0] > 3]

                pos_frequencies_test, neg_frequencies_test = generate_freq(review_training, review_dict)

                combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)

                all_frequencies_test = []

                for i in range(len(combined_testing)):
                    review_rate = combined_testing[i][0]
                    review_text = combined_testing[i][1]
                    if review_text == "":
                        continue

                    try:
                        review_freq = generate_bag_of_words_frequencies(review_dict, [review_text])
                        # Append review_freq to the list
                        all_frequencies_test.append(review_freq)
                    except:
                        print(review_text)

                all_frequencies_test = np.array(all_frequencies_test).T

                percent_filter = 0.2

                mask = calculate_mask(neg_frequencies_test, pos_frequencies_test)
                # filtered_mask, set_to_zero_index = calculate_filter(mask, percent_filter)
                EER_array = []
                far_array = []
                frr_array = []
                ideal_array = []
                percent_filter_array = []
                while percent_filter < 0.4:
                    print(percent_filter)
                    percent_filter_array.append(percent_filter)
                    filtered_mask, set_to_zero_index = calculate_filter(mask, percent_filter)
                    masked_neg = filtered_mask * neg_frequencies_test
                    masked_pos = filtered_mask * pos_frequencies_test
                    # count_poszeros = np.count_nonzero(masked_pos == 0)
                    # count_negzeros = np.count_nonzero(masked_neg == 0)
                    # count_filtered_mask = np.count_nonzero(filtered_mask == 0)
                    # print(count_poszeros, count_negzeros, count_filtered_mask, set_to_zero_index)
                    combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
                    pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)
                    actual_labels = create_actual_labels(combined_testing[:, 0])

                    mid_threshold = calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold,
                                                                    lower_threshold, actual_labels)
                    threshold_matrix = np.arange(mid_threshold - 0.5, mid_threshold + 0.5, 0.002).reshape(-1, 1)

                    far, frr = best_threshold(actual_labels, pos_cosine, neg_cosine, mid_threshold)
                    predicted_labels, EER, thresholds, confusion_matrices, uar_matrix, all_fpr, all_fnr = (
                        confusion_matrix_scheduler(actual_labels, pos_cosine, neg_cosine, threshold_matrix))
                    report = statistics(actual_labels, predicted_labels)
                    print("EER: ", EER)
                    print("FAR: ", far)
                    print("FRR: ", frr)
                    EER_array.append(EER)
                    far_array.append(far)
                    frr_array.append(frr)
                    ideal_array.append(abs(frr - far) + EER)
                    percent_filter += 0.001

                # Plot the data
                plt.plot(percent_filter_array, EER_array, label='EER')
                plt.plot(percent_filter_array, far_array, label='FAR')
                plt.plot(percent_filter_array, frr_array, label='FRR')
                plt.plot(percent_filter_array, ideal_array, label='IDEAL')
                # Add labels and title
                plt.xlabel('Percent Filter')
                plt.ylabel('EER')
                plt.title('Plot of EER over Filter')
                plt.legend()
                # Show grid
                plt.grid(True)

                # Show the plot
                plt.show()

        elif command == 'M':
            print("Executing main function...")

            if (percentage_testing is not None and lower_threshold is not None and upper_threshold is not None
                    and lower_bound is not None and upper_bound is not None and step_value is not None):

                sorted_indices = np.argsort(reviews_matrix[0])
                reviews_matrix = reviews_matrix[:, sorted_indices]

                review_training, review_testing = train_test_split(reviews_matrix.T, test_size=percentage_testing,
                                                                   random_state=42)
                neg_testing = review_testing[review_testing[:, 0] < 3]
                pos_testing = review_testing[review_testing[:, 0] > 3]

                pos_frequencies_test, neg_frequencies_test = generate_freq(review_training, review_dict)

                combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)

                all_frequencies_test = []

                for i in range(len(combined_testing)):
                    review_rate = combined_testing[i][0]
                    review_text = combined_testing[i][1]
                    if review_text == "":
                        continue

                    try:
                        review_freq = generate_bag_of_words_frequencies(review_dict, [review_text])
                        # Append review_freq to the list
                        all_frequencies_test.append(review_freq)
                    except:
                        print(review_text)

                all_frequencies_test = np.array(all_frequencies_test).T

                percent_filter = 0.23

                mask = calculate_mask(neg_frequencies_test, pos_frequencies_test)
                filtered_mask, set_to_zero_index = calculate_filter(mask, percent_filter)

                masked_neg = filtered_mask * neg_frequencies_test
                masked_pos = filtered_mask * pos_frequencies_test
                combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
                pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)

                actual_labels = create_actual_labels(combined_testing[:, 0])

                mid_threshold = calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold,
                                                                lower_threshold,
                                                                actual_labels)
                threshold_matrix = np.arange(mid_threshold - 0.5, mid_threshold + 0.5, 0.002).reshape(-1, 1)

                far, frr = best_threshold(actual_labels, pos_cosine, neg_cosine, mid_threshold)
                predicted_labels, EER, thresholds, confusion_matrices, uar_matrix, all_fpr, all_fnr = (
                    confusion_matrix_scheduler(actual_labels, pos_cosine, neg_cosine, threshold_matrix))
                report = statistics(actual_labels, predicted_labels)
                print("FAR (FPR): ", far)
                print("FRR (FNR): ", frr)


            else:
                print("Hyperparameters are not set. Please set hyperparameters first.")

        elif command == 'G':
            print("Generating graphs...")

            if thresholds is not None and uar_matrix is not None and all_fpr is not None and all_fnr is not None:
                plot_threshold_vs_accuracy(thresholds, uar_matrix)
                calculate_det(all_fpr, all_fnr)
            else:
                print("Graphs cannot be generated. Required data is missing.")

        elif command == 'I':
            print("Setting hyperparameters...")

            percentage_testing = float(input("Enter percentage for testing data (e.g., 0.1 for 10%): "))
            lower_threshold = float(input("Enter lower threshold value (Recommended value=-1): "))
            upper_threshold = float(input("Enter upper threshold value (Recommended value=1): "))
            lower_bound = float(input("Enter lower bound value (Recommended value=-0.1, an integer or float that's "
                                      "less than zero): "))
            upper_bound = float(input("Enter upper bound value (Recommended value=0.1, an integer or float that's "
                                      "greater than zero): "))
            step_value = float(input("Enter step value (Recommended value=0.02): "))
            threshold_values = np.arange(lower_bound, upper_bound, step_value)
            percentage_training = 1 - percentage_testing

            print("Hyperparameters set successfully...")

        elif command == 'P':
            print("Printing data...\n")

            # Print the hyperparameters
            print("Current parameters:")
            if hyperparameters:
                print(tabulate(hyperparameters, headers=["Parameter", "Value"], tablefmt="grid"))
            else:
                print("Hyperparameters are not set.")

            # Check the length of the dictionary
            print("\nLength of the dictionary:", len(review_dict))

            # Print the bag-of-words frequencies if they are set
            if pos_frequencies_train is not None:
                print("\nBag-of-words frequencies (Positive):")
                print(pos_frequencies_train[500:800])
            if neg_frequencies_train is not None:
                print("\nBag-of-words frequencies (Negative):")
                print(neg_frequencies_train[800:1000])

            # Print the frequencies for testing data if they are set
            if pos_frequencies_test is not None:
                print("\nPositive Reviews Frequencies (first 5):")
                print(pos_frequencies_test[:5])
            if neg_frequencies_test is not None:
                print("\nNegative Reviews Frequencies (first 5):")
                print(neg_frequencies_test[:5])

            # Print threshold values and EER if they are set
            if lower_threshold is not None and upper_threshold is not None:
                print("\nThe bounds are:", lower_threshold, "to", upper_threshold)
            if EER is not None:
                print("Equal Error Rate:", EER)

            # Print peak results if available
            if thresholds is not None and confusion_matrices is not None:
                if len(thresholds) > 1:
                    max_index = find_max(uar_matrix)
                    print("\nThe peak results are:\n")
                    print("Threshold:", thresholds[max_index])
                    print("Confusion Matrix:")
                    headers = ["", "Expected Positive", "Expected Negative"]
                    matrix_data = [["", "P", "N"],
                                   ["P", confusion_matrices[max_index][1, 1], confusion_matrices[max_index][1, 0]],
                                   ["N", confusion_matrices[max_index][0, 1], confusion_matrices[max_index][0, 0]]]
                    print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
                    calculate_performance_matrix(confusion_matrices[max_index], 1)
                else:
                    print("\nThe peak results are:\n")
                    print("Threshold:", thresholds[0])
                    print("Confusion Matrix:")
                    headers = ["", "Expected Positive", "Expected Negative"]
                    matrix_data = [["", "P", "N"],
                                   ["P", confusion_matrices[0][1, 1], confusion_matrices[0][1, 0]],
                                   ["N", confusion_matrices[0][0, 1], confusion_matrices[0][0, 0]]]
                    print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
                    calculate_performance_matrix(confusion_matrices[0], 1)
            else:
                print("\nPeak results are not available.")

            # Print the report if it is available
            if report is not None:
                print("\nClassification Report:")
                print(report)

        elif command == 'E':
            print("Program terminated.")
            break

        else:
            print("Invalid command. Please try again.")


main()
