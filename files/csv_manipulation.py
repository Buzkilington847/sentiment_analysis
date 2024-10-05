import csv  # Import the CSV module to handle reading and writing CSV files
import pandas as pd  # Import pandas for working with data structures and reading/writing CSVs
import numpy as np  # Import numpy for array manipulations and numerical operations

# Define the file path to the cleaned review data CSV
REVIEW_DATA_FILEPATH = "../data/reviews/clean_review_data.csv"


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
    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        df.to_csv(file, index=False, header=False)
        file.flush()  # Ensure the file is written immediately


def load_csv(mode, csv_file, index, first_row_header=False):
    """
    Load data from a CSV file, and return a specified row or column.

    Parameters:
        mode (str): Specifies if data is to be loaded as a row ("row") or a column ("column").
        csv_file (str): Path to the CSV file to be loaded.
        index (int): The index of the row or column to return.
        first_row_header (bool): If True, treats the first row of the CSV as a header.

    Returns:
        pandas.Series: The requested row or column as a pandas Series.
    """
    # Read the CSV file, with or without header, based on the input
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')

    # Return the specified row if the mode is "row"
    if mode == "row":
        return df.iloc[index]

    # Return the specified column if the mode is "column"
    elif mode == "column":
        return df.iloc[:, index]

    else:
        print("Invalid mode!")  # Print an error message for invalid mode input


def remove_duplicates_from_csv(input_file, output_file):
    """
    Remove duplicate rows from a CSV file based on the review text in the second column (index 1).
    The cleaned data (without duplicates) is saved to a new CSV file.

    Parameters:
        input_file (str): Path to the input CSV file that contains duplicate rows.
        output_file (str): Path to the output CSV file where cleaned data will be written.

    Returns:
        None
    """
    # Create a set to store unique review texts
    unique_reviews = set()

    # Open the input CSV for reading and output CSV for writing
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='',
                                                                             encoding='utf-8') as outfile:
        reader, writer = csv.reader(infile)  # Create CSV reader and writer
        for row in reader:
            # Strip any surrounding whitespace from the review text (assumed to be in the second column)
            review = row[1].strip()

            # If the review text is not already in the set, add it and write the row to the output file
            if review not in unique_reviews:
                unique_reviews.add(review)
                writer.writerow(row)  # Write the non-duplicate row to the output file

    print(f"Duplicates removed. Cleaned data written to {output_file}.")  # Confirm success


def load_reviews():
    """
    Load the review data from the CSV file and sort the review matrix by the first column.
    Review data structured in many rows, 2 columns. First column is for ratings (float or doubles), second column
    is for review text.

    Returns:
        numpy.ndarray: A matrix of reviews, where rows are sorted based on the first column.
    """
    # Read the CSV file as a numpy array and transpose it (switch rows and columns)
    reviews_matrix = pd.read_csv(REVIEW_DATA_FILEPATH).to_numpy().T

    # Sort the rows of the matrix by the first column and return the sorted matrix
    sorted_indices = np.argsort(reviews_matrix[0])
    reviews_matrix = reviews_matrix[:, sorted_indices]

    return reviews_matrix
