import pandas as pd

def load_cleaned_data(filepath):
    data_df = pd.read_csv(filepath)
    return data_df

def display_summary_statistics(df):
    print("Summary Statistics:")
    print(df.describe())

def display_missing_values(df):
    missing_count_per_column = df.isna().sum()
    print("\nMissing values per column:")
    print(missing_count_per_column)

cleaned_data_filepath = 'data/processed/cleaned_data.csv'

cleaned_data_df = load_cleaned_data(cleaned_data_filepath)

display_summary_statistics(cleaned_data_df)

display_missing_values(cleaned_data_df)
