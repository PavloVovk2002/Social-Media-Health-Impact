import pandas as pd

def load_data(filepath):
    data_df = pd.read_csv(filepath) 
    return data_df

def clean_data(df):
    # Missing Values
    missing_count_per_column = df.isna().sum()
    print("Missing values per column:")
    print(missing_count_per_column)
    return df
