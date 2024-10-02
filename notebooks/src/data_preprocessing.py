import pandas as pd

def load_data(filepath):
  data_df = pd.read_csv('/Users/pavlovovk/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv') 
  return pd.read_csv(filepath)

def clean_data(df):
#Missing Values
  missing_count_per_column = data_df.isna().sum()
  print("Missing values per column:")
  print(missing_count_per_column)
  return df
