import pandas as pd
import os

# Path to the raw data
raw_data = os.path.join('data/raw/South_East_Asia_Social_Media_MentalHealth.csv')

# Load the data
data_df = pd.read_csv(raw_data)

# Check for missing values
missing_values_per_column = data_df.isna().sum()
print("Missing values per column:")
print(missing_values_per_column)

# No missing values so do not need to drop missing values

# Rearrange the columns
data_df = data_df[['Country', 'State', 'Age Group', 'Gender', 'Urban/Rural', 'Education Level', 'Socioeconomic Status',
'Daily SM Usage (hrs)', 'Most Used SM Platform', 'Frequency of SM Use', 'Likes Received (per post)', 'Comments Received (per post)',
'Shares Received (per post)', 'Peer Comparison Frequency (1-10)', 'Cyberbullying Experience (1-10)', 'Anxiety Levels (1-10)',
'Social Anxiety Level (1-10)', 'Body Image Impact (1-10)', 'Sleep Quality Impact (1-10)', 'Self Confidence Impact (1-10)']]

# Path to cleaned data
cleaned_data = os.path.join('data/cleaned/South_East_Asia_Social_Media_MentalHealth_cleaned.csv')

# Generate clean data
data_df.to_csv(cleaned_data, index=False)

print('Cleaned data has been saved to:', cleaned_data)

