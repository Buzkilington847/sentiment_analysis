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

import csv
import numpy as np
import pandas as pd
from num2words import num2words

import csv_manipulation
import dictionary_maker

"""Beginning of Constants"""
REVIEW_DATA_FILEPATH = "../data/reviews/clean_review_data.csv"

DEFAULT_HYPERPARAMETERS = {
    "percentage_testing": 0.1,
    "lower_threshold": -1,
    "upper_threshold": 1,
    "lower_bound": -0.1,
    "upper_bound": 0.1,
    "step_value": 0.02
}
"""End of Constants"""


def get_valid_input(prompt, cast_type=float, condition=lambda x: True, error_message="Invalid input."):
    while True:
        try:
            value = cast_type(input(prompt))
            if condition(value):
                return value
            else:
                print(error_message)
        except ValueError:
            print("Please enter a valid number.")


def setup_hyperparameters(hyperparameters):
    print("Setting hyperparameters...")

    settings = input("Enter Y for default settings or N for custom settings: ").strip().upper()

    if settings == "Y":
        hyperparameters.update(DEFAULT_HYPERPARAMETERS)
    else:
        hyperparameters["percentage_testing"] = get_valid_input(
            "Enter percentage for testing data (e.g., 0.1 for 10%): ",
            float, lambda x: 0.0 < x < 1.0, "Please enter a value between 0 and 1."
        )

        hyperparameters["lower_threshold"] = get_valid_input(
            "Enter lower threshold value (Recommended value=-1): ", float
        )

        hyperparameters["upper_threshold"] = get_valid_input(
            "Enter upper threshold value (Recommended value=1): ",
            float, lambda x: x > hyperparameters["lower_threshold"],
            "Upper threshold must be greater than the lower threshold."
        )

        hyperparameters["lower_bound"] = get_valid_input(
            "Enter lower bound value (Recommended value=-0.1, less than 0): ",
            float, lambda x: x < 0, "Lower bound must be less than 0."
        )

        hyperparameters["upper_bound"] = get_valid_input(
            "Enter upper bound value (Recommended value=0.1, greater than 0): ",
            float, lambda x: x > 0, "Upper bound must be greater than 0."
        )

        hyperparameters["step_value"] = get_valid_input(
            "Enter step value (Recommended value=0.02): ",
            float, lambda x: x > 0, "Step value must be greater than 0."
        )
    percentage_training = 1 - hyperparameters["percentage_testing"]
    return percentage_training


def main_function():
    print("Main function of the program.")
    # Add logic here


def generate_graph():
    print("Generating graph.")
    # Add logic here


def input_hyperparameters(hyperparameters):
    print("Inputting hyperparameters.")
    setup_hyperparameters(hyperparameters)
    # Add logic here


def print_values():
    print("Printing values.")
    # Add logic here


def terminate_program():
    print("Terminating the program.")
    exit()


def percent_filter_array():
    print("Filtering array by percentage.")
    # Add logic here


def squash_array():
    print("Squashing array.")
    # Add logic here


def both_percent_filter_and_squash():
    print("Applying both percent filter and squashing array.")
    # Add logic here


def print_commands():
    print("Commands:")
    print("M - Main function of the program")
    print("G - Graph generation")
    print("I - Input hyperparameters")
    print("P - Print values")
    print("E - Terminate program")
    print("% - Percent Filter Array")
    print("S - Squash Array")
    print("B - Both Percent Filter Array and Squash")
    print("H - Print commands")


def start():
    # Hyperparameters
    hyperparameters = {
        "percentage_testing": None,
        "lower_threshold": None,
        "upper_threshold": None,
        "lower_bound": None,
        "upper_bound": None,
        "step_value": None
    }

    # Create a dictionary mapping commands to functions
    command_dict = {
        "M": main_function,
        "G": generate_graph,
        "I": input_hyperparameters,
        "P": print_values,
        "E": terminate_program,
        "%": percent_filter_array,
        "S": squash_array,
        "B": both_percent_filter_and_squash,
        "H": print_commands,
    }

    # Other variables
    pos_frequencies_train = None
    neg_frequencies_train = None
    pos_frequencies_test = None
    neg_frequencies_test = None
    EER = None
    thresholds = None
    confusion_matrices = None
    report = None
    uar_matrix = None
    all_fpr = None
    all_fnr = None

    # Data loading phase.
    # - Gets n-gram sizes
    # - Generates a dictionary with appropriately sized n-grams
    # - Gets review_data

    gram_size = input("Enter the size for n-gram generation. Example: 2 would generate bigrams: ").strip().upper()

    try:
        # Try to load the n-gram dictionary CSV file
        csv_manipulation.load_csv("row", f"../data/dictionaries/ngram_dictionary_{num2words(gram_size)}.csv", 0)
    except FileNotFoundError:
        # Create the dictionary if the file is not found
        dictionary_maker.create_dictionary(gram_size)

        # Retry loading the CSV after creating the file
        try:
            csv_manipulation.load_csv("row", f"../data/dictionaries/ngram_dictionary_{num2words(gram_size)}.csv", 0)
        except Exception as e:
            print(f"An error occurred while reloading the CSV: {e}")
    except Exception as e:
        # Handle any other exception that occurs during the first CSV load attempt
        print(f"An error occurred: {e}")

    # Try to load the review data, with appropriate error handling
    try:
        review_data = csv_manipulation.load_reviews()
    except FileNotFoundError:
        print("No review data detected!")
    except Exception as e:
        print(f"An error occurred while loading review data: {e}")

    while True:
        # Ensure hyperparameters are set
        if any(value is None for value in hyperparameters.values()):
            percentage_testing = setup_hyperparameters(hyperparameters)

        # Prompt for a command
        print("Press 'H' for a list of commands.")
        command = input("Please enter a command: ").strip().upper()

        # Check if the command exists in the command dictionary and execute the corresponding function
        if command in command_dict:
            # If command is "I", pass the hyperparameters to the function
            if command == "I":
                command_dict[command](hyperparameters)
            else:
                command_dict[command]()
        else:
            print("Invalid command, please try again.")

        break
