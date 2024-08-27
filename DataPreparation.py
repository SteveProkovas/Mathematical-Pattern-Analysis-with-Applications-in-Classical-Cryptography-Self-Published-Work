import numpy as np
import pandas as pd
import argparse
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sum_of_digits(number):
    """
    Calculate the sum of the digits of a given number.

    Args:
    number (int): The number whose digits are to be summed.

    Returns:
    int: The sum of the digits.
    """
    return sum(int(digit) for digit in str(number))

def generate_data(base_numbers, triplets):
    """
    Generate data for the given base numbers and triplets.

    Args:
    base_numbers (list of int): List of base numbers to multiply.
    triplets (list of tuple of int): List of triplets to multiply each base number with.

    Returns:
    pd.DataFrame: The generated dataset containing base number, multiplier, product, and digit sum.
    """
    data = []

    logging.info("Generating data...")
    for base_number in base_numbers:
        for triplet in triplets:
            for multiplier in triplet:
                product = base_number * multiplier
                digit_sum = sum_of_digits(product)
                data.append([base_number, multiplier, product, digit_sum])
                logging.debug(f"Base: {base_number}, Multiplier: {multiplier}, Product: {product}, Digit Sum: {digit_sum}")
    
    # Convert list to DataFrame
    df = pd.DataFrame(data, columns=['BaseNumber', 'Multiplier', 'Product', 'DigitSum'])
    
    return df

def normalize_data(df, method='minmax'):
    """
    Normalize the dataset using Min-Max scaling or Standard scaling.

    Args:
    df (pd.DataFrame): The dataset to be normalized.
    method (str): The normalization method to use ('minmax' or 'standard').

    Returns:
    pd.DataFrame: The normalized dataset.
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Use 'minmax' or 'standard'.")

    scaled_values = scaler.fit_transform(df)
    return pd.DataFrame(scaled_values, columns=df.columns)

def handle_missing_data(df, strategy='mean'):
    """
    Handle missing data in the dataset by imputing values.

    Args:
    df (pd.DataFrame): The dataset with potential missing values.
    strategy (str): The imputation strategy ('mean', 'median', 'most_frequent').

    Returns:
    pd.DataFrame: The dataset with missing values handled.
    """
    imputer = Imputer(strategy=strategy)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns)

def reduce_dimensionality(df, n_components=2):
    """
    Reduce the dimensionality of the dataset using PCA.

    Args:
    df (pd.DataFrame): The dataset to be reduced.
    n_components (int): The number of principal components to retain.

    Returns:
    pd.DataFrame: The reduced dataset.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df)
    return pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

def visualize_data(df):
    """
    Visualize the generated data using basic plots.

    Args:
    df (pd.DataFrame): The dataset to be visualized.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Product'], df['DigitSum'], 'o-', label="Product vs. Digit Sum")
    plt.title("Product vs. Digit Sum")
    plt.xlabel("Product")
    plt.ylabel("Digit Sum")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(args):
    try:
        # Generate the data
        df = generate_data(args.base_numbers, args.triplets)

        # Handle missing data if required
        if args.handle_missing:
            df = handle_missing_data(df, strategy=args.missing_strategy)
            logging.info(f"Missing data handled using {args.missing_strategy} strategy.")

        # Normalize the data if required
        if args.normalize:
            df = normalize_data(df, method=args.norm_method)
            logging.info(f"Data normalized using {args.norm_method} scaling.")

        # Reduce dimensionality if required
        if args.reduce_dim:
            df = reduce_dimensionality(df, n_components=args.n_components)
            logging.info(f"Dimensionality reduced to {args.n_components} components.")

        # Save the data to a file
        df.to_csv(args.output_file, index=False)
        logging.info(f"Data generated and saved as '{args.output_file}'")

        # Optionally visualize the data
        if args.visualize:
            visualize_data(df)
        
        logging.info("Process completed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Command-line arguments parser
    parser = argparse.ArgumentParser(description="Generate and preprocess data based on a mathematical pattern.")
    
    parser.add_argument(
        "--base_numbers",
        nargs='+',
        type=int,
        default=[13, 23, 33, 43],
        help="List of base numbers to use (e.g., 13 23 33)."
    )
    
    parser.add_argument(
        "--triplets",
        nargs='+',
        type=int,
        action='append',
        default=[[11, 12, 13], [14, 15, 16], [17, 18, 19]],
        help="List of triplets to use (e.g., --triplets 11 12 13 --triplets 14 15 16)."
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="pattern_data.csv",
        help="Output file name to save the generated data."
    )
    
    parser.add_argument(
        "--normalize",
        action='store_true',
        help="Normalize the generated data."
    )
    
    parser.add_argument(
        "--norm_method",
        type=str,
        default="minmax",
        help="Normalization method to use ('minmax' or 'standard')."
    )
    
    parser.add_argument(
        "--handle_missing",
        action='store_true',
        help="Handle missing data in the dataset."
    )
    
    parser.add_argument(
        "--missing_strategy",
        type=str,
        default="mean",
        help="Strategy to handle missing data ('mean', 'median', 'most_frequent')."
    )
    
    parser.add_argument(
        "--reduce_dim",
        action='store_true',
        help="Reduce the dimensionality of the dataset using PCA."
    )
    
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of principal components to retain if reducing dimensionality."
    )
    
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Visualize the generated data using basic plots."
    )
    
    args = parser.parse_args()
    
    # Execute main function
    main(args)
