import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data_df = pd.read_csv('/Users/pavlovovk/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv')
#data_df = pd.read_csv('/Users/gopivaghani/Documents)
#data_df = pd.read_csv('/Users/amaankhan/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv')

#Clean

#Missing Values
missing_count_per_column = data_df.isna().sum()

print("Missing values per column:")
print(missing_count_per_column)


#Plots

#1 What is the distribution of daily social media usage across different age groups?
#   sns.boxplot(x='Age Group', y='Daily SM Usage (hrs)', data=data_df)
#   plt.title('Daily SM Usage Across Age Groups')

#2 Which platform is the most used by gender?
plt.figure(figsize=(12, 6))
sns.countplot(x='Most Used SM Platform', 
              hue='Gender', 
              data=data_df, 
              palette='pastel') 

plt.title('Most Used SM Platform by Gender', fontsize=16)
plt.xlabel('Social Media Platform', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.xlim(-0.5, len(data_df['Most Used SM Platform'].unique()) - 0.5)
plt.ylim(33000, 35000)

plt.grid(axis='y')
plt.legend(title='Gender')
plt.show()

#3 How do daily social media usage hours vary between urban and rural areas?
#   sns.boxplot(x='Urban/Rural', y='Daily SM Usage (hrs)', data=data_df)
#   plt.title('Daily SM Usage in Urban vs. Rural Areas')

#4 Which country has the highest average social anxiety level?
#   data_df.groupby('Country')['Social Anxiety Level (1-10)'].mean().sort_values().plot(kind='bar')
#   plt.title('Average Social Anxiety Level by Country')

#5 Is there a relationship between peer comparison frequency and social anxiety?
data_df = data_df.sort_values(by='Peer Comparison Frequency (1-10)')
plt.figure(figsize=(10, 6))
sns.lineplot(x='Peer Comparison Frequency (1-10)', 
             y='Social Anxiety Level (1-10)', 
             data=data_df, 
             marker='o')
plt.title('Relationship Between Peer Comparison Frequency and Social Anxiety Level', fontsize=16)
plt.xlabel('Peer Comparison Frequency (1-10)', fontsize=14)
plt.ylabel('Social Anxiety Level (1-10)', fontsize=14)
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

#6 How does the frequency of cyberbullying vary by country?
#   sns.boxplot(x='Country', y='Cyberbullying Experience (1-10)', data=data_df)
#   plt.title('Cyberbullying Experience by Country')
#   plt.xticks(rotation=90)

#7 What is the average self-confidence impact score across different age groups?
#   data_df.groupby('Age Group')['Self Confidence Impact (1-10)'].mean().plot(kind='bar')
#   plt.title('Average Self Confidence Impact by Age Group')

#8 What is the relationship between anxiety levels and sleep quality impact?
#   sns.scatterplot(x='Anxiety Levels (1-10)', y='Sleep Quality Impact (1-10)', data=data_df)
#   plt.title('Anxiety Levels vs. Sleep Quality Impact')

#9 Which platform has the most impact on self-confidence levels?
#   sns.boxplot(x='Most Used SM Platform', y='Self Confidence Impact (1-10)', data=data_df)
#   plt.title('Self Confidence Impact by Platform')

#10 How does peer comparison frequency affect self-confidence?
#   sns.scatterplot(x='Peer Comparison Frequency (1-10)', y='Self Confidence Impact (1-10)', data=data_df)
#   plt.title('Peer Comparison Frequency vs. Self Confidence Impact')

#11 How does the most used social media platform differ between various countries?
#   sns.boxplot(x='Country', y='Most Used SM Platform', data=data_df)
#   plt.title('Most Used Social Media Platform by Country')

#12 How does the number of likes received on a post impact the peer comparison frequency?
#   sns.scatterplot(x='Likes Received (per post)', y='Peer Comparsion Frequency (1-10)', data=data_df)
#   plt.title('Likes Received vs. Peer Comparsion Frequency')

#13 How does the frequency of social media usage correlate with the age groups?
#   sns.scatterplot(x='Frequency of SM Use', y='Age Group', data=data_df)
#   plt.title('Frequency Social Media Usage vs. Age Groups')

#14 What is the relationship between comments received on a post and self confidence level?
#   sns.scatterplot(x='Comments Received (per post)', y='Self Confidenece Impact (1-10)', data=data_df)
#   plt.title('Comments Received vs. Self Confidence Impact')

#15 Which state has the highest cyberbullying experience?
#   data_df.groupby('State')['Cyberbullying Experience'].mean().sort_values().plot(kind='bar')
#   plt.title('Cyberbullying Experience by State')

#16 How does the average social anxiety level differ between age groups?
#   sns.boxplot(x='Age Group', y='Social Anxiety Level (1-10)', data=data_df)
#   plt.title('Social Anxiety Level by Age Groups')

#17 What is the distribution of the most used social media platform between urban and rural areas?
#   sns.boxplot(x='Urban/Rural', y='Most Used SM Platform', data=data_df)
#   plt.title('Most Used Social Media Platform in Urban vs. Rural')

#18 How does cyberbullying experience impact the self confidence level?
#   sns.scatterplot(x='Cyberbullying Experience', y='Self Confidence Impact (1-10)', data=data_df)
#   plt.title('Cyberbullying Experience vs Self Confidence Impact')

#19  What is the relationship between peer comparison frequency and sleep quality impact?
#    sns.scatterplot(x='Peer Comparison Frequency', y='Sleep Quality Impact(1-10)', data=data_df)
#    plt.title('Peer Comparison Frequency vs Sleep Quality Impact')

#20 What are the average anxiety levels across various age groups?
average_anxiety = data_df.groupby('Age Group')['Anxiety Levels (1-10)'].mean()

plt.figure(figsize=(12, 6))
average_anxiety.plot(kind='bar', color='purple', edgecolor='purple')

plt.title('Average Anxiety Levels by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Average Anxiety Level (1-10)', fontsize=14)

plt.ylim(0, 10)
plt.xlim(-0.5, len(average_anxiety) - 0.5)

plt.grid(axis='y', linestyle='--', alpha=0.7)
for index, value in enumerate(average_anxiety):
    plt.text(index, value + 0.1, f'{value:.2f}', ha='center', fontsize=10)

plt.xticks(rotation=45) 
plt.tight_layout()
plt.show()

#21 What is the correlation between socioeconomic status and social anxiety levels?
#   sns.boxplot(x='Socioeconomic Status', y='Social Anxiety Level (1-10)', data=data_df)
#   plt.title('Social Anxiety Level by Socioeconomic Status')

#22 Does daily social media usage vary significantly between different education levels?
#   sns.boxplot(x='Education Level', y='Daily SM Usage (hrs)', data=data_df)
#   plt.title('Daily Social Media Usage by Education Level')

#23 How do peer comparison frequencies differ across countries?
#   sns.boxplot(x='Country', y='Peer Comparison Frequency (1-10)', data=data_df)
#   plt.xticks(rotation=90)
#   plt.title('Peer Comparison Frequency by Country')

#24 What is the relationship between the number of likes received and social anxiety levels?
#   sns.scatterplot(x='Likes Received (per post)', y='Social Anxiety Level (1-10)', data=data_df)
#   plt.title('Likes Received vs Social Anxiety Level')

#25 How does body image impact scores vary across different age groups and genders?
#   sns.boxplot(x='Age Group', y='Body Image Impact (1-10)', hue='Gender', data=data_df)
#   plt.title('Body Image Impact by Age Group and Gender')

#26 Is there a significant difference in sleep quality impact between users who frequently experience cyberbullying and those who don’t?
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cyberbullying Experience (1-10)', y='Sleep Quality Impact (1-10)', hue='Gender', data=data_df, palette='Set2')
plt.title('Sleep Quality Impact by Cyberbullying Experience', fontsize=16)
plt.xlabel('Cyberbullying Experience (1-10)', fontsize=14)
plt.ylabel('Sleep Quality Impact (1-10)', fontsize=14)

plt.xlim(-1, 10)
plt.ylim(0, 11)

plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(title='Gender')
plt.tight_layout()
plt.show()

#27 How does the frequency of social media usage impact self-confidence across different socioeconomic statuses?
#   sns.boxplot(x='Socioeconomic Status', y='Self Confidence Impact (1-10)', data=data_df)
#   plt.title('Self Confidence Impact by Socioeconomic Status')

#28 Which country has the highest body image impact score?
#   sns.boxplot(x='Country', y='Body Image Impact (1-10)', data=data_df)
#   plt.xticks(rotation=90)
#   plt.title('Body Image Impact by Country')

#29 How does the number of comments received on posts affect social anxiety levels?
#   sns.scatterplot(x='Comments Received (per post)', y='Social Anxiety Level (1-10)', data=data_df)
#   plt.title('Comments Received vs Social Anxiety Level')

#30 What is the average social anxiety levels by number of comments recieved?
bins = [0, 5, 10, 20, 50, 100, 200]
labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101-200']
data_df['Comments Binned'] = pd.cut(data_df['Comments Received (per post)'], bins=bins, labels=labels)

average_anxiety = data_df.groupby('Comments Binned')['Social Anxiety Level (1-10)'].mean()

plt.figure(figsize=(8, 8))
plt.pie(average_anxiety, labels=average_anxiety.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
plt.title('Average Social Anxiety Levels by Number of Comments Received', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()