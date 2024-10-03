import pandas as pd
from src.utils import load_data, clean_data

raw_data_filepath = 'data/raw/South_East_Asia_Social_Media_MentalHealth.csv'
processed_data_filepath = 'data/processed/cleaned_data.csv'

data_df = load_data(raw_data_filepath)

cleaned_data = clean_data(data_df)

cleaned_data.to_csv(processed_data_filepath, index=False)

print('Cleaned data has been saved to:', processed_data_filepath)
