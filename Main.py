import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data_df = pd.read_csv('/Users/pavlovovk/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv')
#data_df = pd.read_csv
#data_df = pd.read_csv

#Clean

#Missing Values
missing_count_per_column = data_df.isna().sum()

print("Missing values per column:")
print(missing_count_per_column)


#Plots

#1 What is the distribution of daily social media usage across different age groups?
sns.boxplot(x='Age Group', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Daily SM Usage Across Age Groups')

#2 Which platform is the most used by gender?
sns.countplot(x='Most Used SM Platform', hue='Gender', data=data_df)
plt.title('Most Used SM Platform by Gender')

#3 How do daily social media usage hours vary between urban and rural areas?
sns.boxplot(x='Urban/Rural', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Daily SM Usage in Urban vs. Rural Areas')

#4 Which country has the highest average social anxiety level?
data_df.groupby('Country')['Social Anxiety Level (1-10)'].mean().sort_values().plot(kind='bar')
plt.title('Average Social Anxiety Level by Country')

#5 Is there a relationship between peer comparison frequency and social anxiety?
sns.scatterplot(x='Peer Comparison Frequency (1-10)', y='Social Anxiety Level (1-10)', data=data_df)
plt.title('Peer Comparison Frequency vs. Social Anxiety Level')

#6 How does the frequency of cyberbullying vary by country?
sns.boxplot(x='Country', y='Cyberbullying Experience (1-10)', data=data_df)
plt.title('Cyberbullying Experience by Country')
plt.xticks(rotation=90)

#7 What is the average self-confidence impact score across different age groups?
data_df.groupby('Age Group')['Self Confidence Impact (1-10)'].mean().plot(kind='bar')
plt.title('Average Self Confidence Impact by Age Group')

#8 What is the relationship between anxiety levels and sleep quality impact?
sns.scatterplot(x='Anxiety Levels (1-10)', y='Sleep Quality Impact (1-10)', data=data_df)
plt.title('Anxiety Levels vs. Sleep Quality Impact')

#9 Which platform has the most impact on self-confidence levels?
sns.boxplot(x='Most Used SM Platform', y='Self Confidence Impact (1-10)', data=data_df)
plt.title('Self Confidence Impact by Platform')

#10 How does peer comparison frequency affect self-confidence?
sns.scatterplot(x='Peer Comparison Frequency (1-10)', y='Self Confidence Impact (1-10)', data=data_df)
plt.title('Peer Comparison Frequency vs. Self Confidence Impact')
