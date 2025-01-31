from pathlib import Path
import random
import pandas as pd
import numpy as np
from scipy.stats import skew
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import KFold
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils import parse_yaml


class Splitter:
    def __init__(
        self,
        toc_path: str,
        seed: int,
        folds: int,
        alpha_p_combinations: list[list[float]],
        num_centres: int,
        num_classes: list[int],
        csv_path: str,
        bubbleplot_path: str,
    ) -> None:
        self.toc_path = Path(toc_path)
        self.seed = seed
        self.folds = folds
        self.alpha_p_combinations: list[tuple] = [
            tuple(comb) for comb in alpha_p_combinations
        ]
        self.num_centres = num_centres
        self.cardinalities = num_classes
        self.csv_path = Path(csv_path)
        self.bubbleplot_path = Path(bubbleplot_path)

        self.seed_everything()

    def seed_everything(self):
        """
        Seed all the relevant random generators for reproducibility.
        """
        random.seed(self.seed)

    def read_toc(self):
        """
        Load ToC (csv file) of chosen dataset as a pandas dataframe.
        """
        self.toc = pd.read_csv(self.toc_path, sep=";")

    def label_conversion_rule(self, label, cardinality):
        if cardinality == 2:
            label = 0 if label < 4 else 1
        elif cardinality == 4:
            if label < 3:
                label = 1
            elif label < 5:
                label = 2
            elif label < 7:
                label = 3
            else:
                label = 4
        elif cardinality == 6:
            if label < 3:
                label = 1
            elif label < 7:
                label = label - 1
            else:
                label = 6
        elif cardinality == 8:
            # No change needed for 8 classes
            label = label
        return label

    def split_dirichlet_bernoulli(
        self, df: pd.DataFrame, alpha_dir, p_bernoulli, heterogeneity_factor
    ):
        """
        Args:
        - df: A pandas DataFrame with at least two columns: 'filename' and 'label'
        - num_clients: The number of clients
        - alpha_dir: Parameter for the Dirichlet distribution
        - p_bernoulli: Probability for Bernoulli distribution

        Returns:
        - client_data: Dictionary where each key is a client and each value is a list of filenames assigned to that client
        """

        # Get the unique class labels from the 'label' column
        classes = sorted(list(df["Label"].unique()))
        num_classes = len(classes)

        # Initialize the structure to hold client data
        client_data = defaultdict(list)

        # Step 1: Weighted Bernoulli Sampling with Safeguard
        # Create a probability array that decreases for higher classes
        bernoulli_probs = [
            p_bernoulli * (1 - heterogeneity_factor * (i / num_classes))
            for i in range(num_classes)
        ]
        bernoulli_matrix = np.random.binomial(
            1, bernoulli_probs, size=(self.num_centres, num_classes)
        )

        # Ensure that each class belongs to at least one client
        for class_idx in classes:
            if np.sum(bernoulli_matrix[:, class_idx]) == 0:
                # Force one client to own this class
                client_idx = np.random.randint(self.num_centres)
                bernoulli_matrix[client_idx, class_idx] = 1

        # Step 2: Distribute data for each class
        for class_idx in classes:
            # Get all data samples (filenames) for this class
            filenames = df[df["Label"] == class_idx]["Filename"].values
            np.random.shuffle(filenames)
            # Find which clients own samples of this class
            owning_clients = np.where(bernoulli_matrix[:, class_idx] == 1)[0]

            # Step 3: Dirichlet distribution for the clients that own this class
            if len(owning_clients) > 0:
                # Adjust alpha for upper classes for increased heterogeneity
                adjusted_alpha = alpha_dir * (
                    1 - heterogeneity_factor * (class_idx / num_classes)
                )
                dirichlet_proportions = np.random.dirichlet(
                    [alpha_dir] * len(owning_clients)
                )

                # Split the class data according to the Dirichlet proportions
                split_sizes = (
                    (dirichlet_proportions * len(filenames)).round().astype(int)
                )

                # Make sure no samples are left unassigned due to rounding
                if leftovers := len(filenames) - np.sum(split_sizes):
                    print(f"extra samples {leftovers=} slapped into last client")
                    split_sizes[-1] += leftovers

                # Assign data to clients
                data_splits = np.split(filenames, np.cumsum(split_sizes)[:-1])
                for client_idx, data_split in zip(owning_clients, data_splits):
                    client_data[client_idx].extend(data_split)

        return client_data

    def calculate_heterogeneity_factor(self, alpha, p, alpha_max, p_max):
        """
        Calculate a heterogeneity factor based on alpha and p, normalized by max values.
        The lower the values of alpha and p, the higher the heterogeneity factor.
        """
        return (1 - (alpha / alpha_max)) * 0.5 + (1 - (p / p_max)) * 0.5

    def convert_labels(self, cardinality):
        """
        Function to adjust labels in a copy of the DataFrame based on the class cardinality
        and handle the case where the labels are 1-indexed instead of the desired 0-indexing.

        Returns:
        - pd.DataFrame: A new DataFrame with the updated "label" column (leaving the original unchanged).
        """
        # self.read_toc()
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = self.toc.copy()

        # Check the highest value in the "label" column
        # max_label = df_copy["Label"].max()

        # Check the minimum value in the "label" column
        min_label = df_copy["Label"].min()

        # # If the highest label value is less than or equal to the number of classes, no conversion needed
        # if max_label < cardinality:
        #     print("No conversion needed as the labels already fit the desired class cardinality.")
        #     return df_copy

        # Apply the conversion logic to the "label" column of the copy
        df_copy["Label"] = df_copy["Label"].apply(
            lambda label: self.label_conversion_rule(label, cardinality)
        )

        # Check if the minimum label is 1, and if so, convert the labels to label-1
        if min_label == 1:
            df_copy["Label"] = df_copy["Label"] - 1

        # Return the new DataFrame with updated labels
        return df_copy

    def plot_bubble_distribution(self, client_data, df, plot_name):
        """
        Args:
        - client_data: Dictionary where each key is a client and value is the list of filenames assigned to that client
        - df: Original dataframe with columns 'filename' and 'label'

        This function plots a bubble plot showing the distribution of classes across clients.
        """
        classes = sorted(list(df["Label"].unique()))
        num_classes = len(classes)

        # Prepare a matrix where rows represent clients and columns represent classes
        distribution_matrix = np.zeros((self.num_centres, num_classes))

        # Fill the matrix with the count of samples for each client and class
        for client_id, filenames in client_data.items():
            for filename in filenames:
                label = df.loc[df["Filename"] == filename, "Label"].values[0]
                class_idx = np.where(classes == label)[0][
                    0
                ]  # Find the index of the class
                distribution_matrix[client_id, class_idx] += 1

        # Plot the bubble plot
        fig, ax = plt.subplots()

        # Create a scatter plot with varying bubble sizes
        for client_id in range(self.num_centres):
            for class_idx in range(num_classes):
                size = distribution_matrix[client_id, class_idx]
                if size > 0:
                    ax.scatter(
                        class_idx, client_id, s=size * 5, c="blue", alpha=0.6
                    )  # s=size*10 controls bubble size

        ax.set_xlabel("Class ID")
        ax.set_ylabel("Client ID")
        ax.set_title("Distribution of Samples Across Clients and Classes")

        # Set ticks for x and y axes
        ax.set_xticks(np.arange(num_classes))
        ax.set_xticklabels(classes)  # Use sorted class labels for the x-axis
        ax.set_yticks(np.arange(self.num_centres))
        # Add grid beneath the scatter plot
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        self.bubbleplot_path.mkdir(exist_ok=True, parents=True)

        plt.savefig(self.bubbleplot_path / f"{plot_name}.pdf")
        plt.close()

    def process_and_save_splits(self):
        self.read_toc()

        alpha_max = max(alpha for alpha, _ in self.alpha_p_combinations)
        p_max = max(p for _, p in self.alpha_p_combinations)

        for cardin in self.cardinalities:
            df_converted = self.convert_labels(cardin)
            new_columns = {}

            for alpha, p in self.alpha_p_combinations:
                # Calculate heterogeneity factor for this alpha and p combination
                heterogeneity_factor = self.calculate_heterogeneity_factor(
                    alpha, p, alpha_max, p_max
                )
                for fold_num in range(self.folds):
                    np.random.seed(fold_num)
                    split_results: dict[int, list[str]] = (
                        self.split_dirichlet_bernoulli(
                            df_converted, alpha, p, heterogeneity_factor
                        )
                    )
                    col_name = f"Fold_{fold_num}_Alpha_{alpha}_P_{p}"
                    centre_assignments = [""] * len(df_converted)
                    for client, filenames in split_results.items():
                        index = df_converted.index[
                            df_converted["Filename"].isin(filenames)
                        ]

                        for i in index:
                            centre_assignments[i] = client

                    new_columns[col_name] = centre_assignments
                    print(f"Saved column: {col_name}")

                    plot_name = f"Card_{cardin}_{col_name}"

                    self.plot_bubble_distribution(
                        split_results, df_converted, plot_name
                    )

            df_new_columns = pd.DataFrame(new_columns, index=df_converted.index)
            df_converted = pd.concat([df_converted, df_new_columns], axis=1)
            filename = f"classification_{cardin}_classes.csv"
            self.csv_path.mkdir(exist_ok=True, parents=True)
            df_converted.to_csv(self.csv_path / filename, index=False)
            print(f"Saved CSV for {cardin} classes: {filename}")

    def run(self):
        """
        Run the entire pipeline: generate splits and save them to files.
        """
        self.process_and_save_splits()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, help="Name of dataset to use; available datasets: CSAW-M"
    )
    args = parser.parse_args()
    current_dir = Path(__file__).parent
    splitter_config_path = current_dir.parent / "conf" / "split_data.yaml"
    config = parse_yaml(splitter_config_path)
    (
        toc_path,
        seed,
        folds,
        alpha_p_combinations,
        num_centres,
        num_classes,
        csv_path,
        bubbleplot_path,
    ) = (
        config[args.dataset][key]
        for key in [
            "toc_path",
            "seed",
            "folds",
            "alpha_p_combinations",
            "num_centres",
            "num_classes",
            "csv_path",
            "bubbleplot_path",
        ]
    )
    splitter = Splitter(
        toc_path,
        seed,
        folds,
        alpha_p_combinations,
        num_centres,
        num_classes,
        csv_path,
        bubbleplot_path,
    )
    splitter.run()

    # Run tests on generated CSV files
    original_df = pd.read_csv(toc_path, sep=";")

    for num_class in num_classes:
        # The generated CSV for the current class cardinality
        csv_file = Path(csv_path) / f"classification_{num_class}_classes.csv"

        # Run tests on the generated CSV files
        print(f"\nRunning tests on: {csv_file}")

        # Test if the total number of samples matches the original dataset
        test_total_samples(csv_file, original_df)

        # Test class distribution matches after label conversion
        test_class_distribution(
            csv_file, original_df, num_class, splitter.label_conversion_rule
        )

        # Test that each sample is assigned to exactly one center per fold and (alpha, p) combination
        test_single_center_assignment_per_sample(csv_file, folds, alpha_p_combinations)

        test_missing_values(csv_file, alpha_p_combinations, folds)

    print("All tests passed for all CSV files!")


def test_total_samples(csv_path, original_df):
    df = pd.read_csv(csv_path)
    original_sample_count = original_df.shape[0]
    csv_sample_count = df["Filename"].nunique()
    assert (
        original_sample_count == csv_sample_count
    ), f"Mismatch: {original_sample_count} vs {csv_sample_count}"
    print("Test passed: Total number of samples matches the original dataset.")


def test_class_distribution(csv_path, original_df, num_classes, conversion_function):
    print(f"\nRunning class distribution test for {num_classes} classes...")

    # Load the processed CSV
    df = pd.read_csv(csv_path)

    # Apply the conversion function to the original class labels
    converted_labels = original_df["Label"].apply(
        lambda x: conversion_function(x, num_classes) - 1
    )

    # Get class distributions (value counts) in the original dataset
    original_class_counts = converted_labels.value_counts().to_dict()
    print("Original class distribution (after conversion):", original_class_counts)

    # Get class distributions in the CSV
    csv_class_counts = df["Label"].value_counts().to_dict()
    print("CSV class distribution:", csv_class_counts)

    # Print both distributions side by side for comparison
    for key in sorted(set(original_class_counts.keys()).union(csv_class_counts.keys())):
        print(
            f"Class {key}: Original = {original_class_counts.get(key, 0)}, CSV = {csv_class_counts.get(key, 0)}"
        )

    # Perform the test
    assert original_class_counts == csv_class_counts, "Mismatch in class distributions."
    print(f"Test passed: Class distribution for {num_classes} classes is correct.\n")


def test_single_center_assignment_per_sample(csv_path, num_folds, alpha_p_combinations):
    """
    Test to check that each sample is assigned to exactly one center per fold and (alpha, p) combination.

    Args:
    - csv_path: Path to the CSV file to be tested.
    - num_folds: The number of folds (e.g., 5).
    - alpha_p_combinations: List of (alpha, p) combinations used during the experiment.

    Raises:
    - AssertionError: If any sample is not assigned to exactly one center per fold and (alpha, p) combination.
    """
    # Load the processed CSV file
    df = pd.read_csv(csv_path)

    # Iterate over each (alpha, p) combination
    for alpha, p in alpha_p_combinations:
        # Check for each fold
        for fold_num in range(num_folds):
            # Create the column name pattern for this fold and (alpha, p) combination
            col_name = f"Fold_{fold_num}_Alpha_{alpha}_P_{p}"

            # Ensure that the column exists in the DataFrame
            assert col_name in df.columns, f"Column {col_name} missing from CSV."

            # Check that each sample has exactly one center assigned
            for index, row in df.iterrows():
                center = row[col_name]

                # Ensure that the center is not empty
                assert (
                    center != ""
                ), f"Sample {row['Filename']} has no center assigned in column {col_name}."

                # Optionally, you could check that the center is valid (for example, an integer or within a valid range)
                # For example, assuming center is a valid integer center ID between 0 and `num_clients - 1`:
                # assert isinstance(center, int), f"Center value {center} is not an integer for sample {row['Filename']}."
                # assert 0 <= int(center) < num_clients, f"Center {center} for sample {row['Filename']} is out of range."

    print(
        "Test passed: Each sample is assigned to exactly one center per fold and (alpha, p) combination."
    )


def test_missing_values(csv_path, alpha_p_combinations, num_folds):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Check if there are any missing values in the "Filename" column
    assert df["Filename"].notna().all(), "There are missing Filenames in the CSV."

    # Find all columns that match the naming pattern for center assignments (Fold_{fold_num}_Alpha_{alpha}_P_{p})
    center_cols = [
        col
        for col in df.columns
        for alpha, p in alpha_p_combinations
        for fold_num in range(num_folds)
        if f"Fold_{fold_num}_Alpha_{alpha}_P_{p}" in col
    ]

    # Check if there are missing values in any of the center assignment columns
    assert (
        df[center_cols].notna().all().all()
    ), "There are missing center assignments in the CSV."

    print("Test passed: No missing values in the CSV.")


if __name__ == "__main__":
    main()
