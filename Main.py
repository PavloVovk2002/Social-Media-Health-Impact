import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#data_df = pd.read_csv('/Users/pavlovovk/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv')
data_df = pd.read_csv('/Users/gopivaghani/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv')
#data_df = pd.read_csv


#Clean
data_df = data_df.drop(columns=['Likes Received (per post)'])

print(data_df.head())


#Plots

# What is the distribution of daily social media usage across different age groups?
sns.boxplot(x='Age Group', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Daily SM Usage Across Age Groups')
plt.show()

# Which platform is the most used by gender?
sns.countplot(x='Most Used SM Platform', hue='Gender', data=data_df)
plt.title('Most Used SM Platform by Gender')
plt.show()

# How do daily social media usage hours vary between urban and rural areas?
sns.boxplot(x='Urban/Rural', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Daily SM Usage in Urban vs. Rural Areas')
plt.show()

# Which country has the highest average social anxiety level?
data_df.groupby('Country')['Social Anxiety Level (1-10)'].mean().sort_values().plot(kind='bar')
plt.title('Average Social Anxiety Level by Country')
plt.show()

# Is there a relationship between peer comparison frequency and social anxiety?
sns.scatterplot(x='Peer Comparison Frequency (1-10)', y='Social Anxiety Level (1-10)', data=data_df)
plt.title('Peer Comparison Frequency vs. Social Anxiety Level')
plt.show()

# How does the frequency of cyberbullying vary by country?
sns.boxplot(x='Country', y='Cyberbullying Experience (1-10)', data=data_df)
plt.title('Cyberbullying Experience by Country')
plt.xticks(rotation=90)
plt.show()

# What is the average self-confidence impact score across different age groups?
data_df.groupby('Age Group')['Self Confidence Impact (1-10)'].mean().plot(kind='bar')
plt.title('Average Self Confidence Impact by Age Group')
plt.show()
