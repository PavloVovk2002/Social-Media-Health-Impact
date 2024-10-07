# script1.py

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

clean_data = os.path.join('data/cleaned/South_East_Asia_Social_Media_Health_cleaned.csv')
data_df = pd.read_csv(clean_data)

#1 What is the distribution of daily social media usage across different age groups?, 3, 22
sns.boxplot(x='Age Group', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Daily SM Usage Across Age Groups')

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

#3 How do daily social media usage hours vary between urban and rural areas?
sns.boxplot(x='Urban/Rural', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Daily SM Usage in Urban vs. Rural Areas')

#4 Which country has the highest average social anxiety level?
data_df.groupby('Country')['Social Anxiety Level (1-10)'].mean().sort_values().plot(kind='bar')
plt.title('Average Social Anxiety Level by Country')

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

#11 How does the most used social media platform differ between various countries?
sns.boxplot(x='Country', y='Most Used SM Platform', data=data_df)
plt.title('Most Used Social Media Platform by Country')

#12 How does the number of likes received on a post impact the peer comparison frequency?
sns.scatterplot(x='Likes Received (per post)', y='Peer Comparison Frequency (1-10)', data=data_df)
plt.title('Likes Received vs. Peer Comparsion Frequency')

#13 How does the frequency of social media usage correlate with the age groups?
sns.scatterplot(x='Frequency of SM Use', y='Age Group', data=data_df)
plt.title('Frequency Social Media Usage vs. Age Groups')

#14 What is the relationship between comments received on a post and self confidence level?
sns.scatterplot(x='Comments Received (per post)', y='Self Confidence Impact (1-10)', data=data_df)
plt.title('Comments Received vs. Self Confidence Impact')

#15 Which state has the highest cyberbullying experience?
data_df.groupby('State')['Cyberbullying Experience (1-10)'].mean().sort_values().plot(kind='bar')
plt.title('Cyberbullying Experience by State')

#16 How does the average social anxiety level differ between age groups?
sns.boxplot(x='Age Group', y='Social Anxiety Level (1-10)', data=data_df)
plt.title('Social Anxiety Level by Age Groups')

#17 What is the distribution of the most used social media platform between urban and rural areas?
sns.boxplot(x='Urban/Rural', y='Most Used SM Platform', data=data_df)
plt.title('Most Used Social Media Platform in Urban vs. Rural')

#18 How does cyberbullying experience impact the self confidence level?
sns.scatterplot(x='Cyberbullying Experience (1-10)', y='Self Confidence Impact (1-10)', data=data_df)
plt.title('Cyberbullying Experience vs Self Confidence Impact')

#19  What is the relationship between peer comparison frequency and sleep quality impact?
sns.scatterplot(x= 'Peer Comparison Frequency (1-10)', y='Sleep Quality Impact (1-10)', data=data_df)
plt.title('Peer Comparison Frequency vs Sleep Quality Impact')

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

#21 What is the correlation between socioeconomic status and social anxiety levels?
sns.boxplot(x='Socioeconomic Status', y='Social Anxiety Level (1-10)', data=data_df)
plt.title('Social Anxiety Level by Socioeconomic Status')

#22 Does daily social media usage vary significantly between different education levels?
sns.boxplot(x='Education Level', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Daily Social Media Usage by Education Level')

#23 How do peer comparison frequencies differ across countries?
sns.boxplot(x='Country', y='Peer Comparison Frequency (1-10)', data=data_df)
plt.xticks(rotation=90)
plt.title('Peer Comparison Frequency by Country')

#24 What is the relationship between the number of likes received and social anxiety levels?
sns.scatterplot(x='Likes Received (per post)', y='Social Anxiety Level (1-10)', data=data_df)
plt.title('Likes Received vs Social Anxiety Level')

#25 How does body image impact scores vary across different age groups and genders?
sns.boxplot(x='Age Group', y='Body Image Impact (1-10)', hue='Gender', data=data_df)
plt.title('Body Image Impact by Age Group and Gender')

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

#27 How does the frequency of social media usage impact self-confidence across different socioeconomic statuses?
sns.boxplot(x='Socioeconomic Status', y='Self Confidence Impact (1-10)', data=data_df)
plt.title('Self Confidence Impact by Socioeconomic Status')

#28 Which country has the highest body image impact score?
sns.boxplot(x='Country', y='Body Image Impact (1-10)', data=data_df)
plt.xticks(rotation=90)
plt.title('Body Image Impact by Country')

#29 How does the number of comments received on posts affect social anxiety levels?
sns.scatterplot(x='Comments Received (per post)', y='Social Anxiety Level (1-10)', data=data_df)
plt.title('Comments Received vs Social Anxiety Level')

#30 What is the average social anxiety levels by number of comments recieved?
bins = [0, 5, 10, 20, 50, 100, 200]
labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101-200']
data_df['Comments Binned'] = pd.cut(data_df['Comments Received (per post)'], bins=bins, labels=labels)

average_anxiety = data_df.groupby(data_df['Comments Binned'])['Social Anxiety Level (1-10)'].mean()

plt.figure(figsize=(8, 8))
plt.pie(average_anxiety, labels=average_anxiety.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
plt.title('Average Social Anxiety Levels by Number of Comments Received', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

#30 What is the relationship between anxiety levels and the number of shares received per post?
sns.scatterplot(x='Shares Received (per post)', y='Anxiety Levels (1-10)', data=data_df)
plt.title('Shares Received vs Anxiety Levels')


#Part 2

#1  What are the statistics for the data? 
statistics_summary = data_df.describe()
print(statistics_summary)

#2  What is the most used social media platform?
most_used_social_media_platform = data_df['Most Used SM Platform'].describe()
print('Most Commonly Used Social Media Platform:')
print(most_used_social_media_platform)

#3  How does the average number of shares per post differ across states? 
sns.lineplot(x='Shares Received (per post)', y='State', data=data_df)
plt.title('Shares Received on post by state')

#4  What age group experience peer comparison the most?
plt.figure(figsize=(10, 6))
sns.boxplot(x='State', y='Body Image Impact (1-10)', data=data_df)
plt.title('Body Image Impact by State and Region')
plt.xlabel('State')
plt.ylabel('Body Image Impact (1-10)')

#5  How often does a person with a certain socioeconomic status use social media? 
plt.figure(figsize=(10, 6))
sns.boxplot(x='State', y='Sleep Quality Impact (1-10)', data=data_df)
plt.title('State-wise Distribution of Sleep Quality Impact')
plt.xlabel('State')

#6  How do education levels within different states affect social media usage patterns? 
plt.figure(figsize=(10, 6))
sns.boxplot(x='Country', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Social Media Addiction Rates by Country')
plt.xlabel('Country')
plt.ylabel('Daily Social Media Usage (hrs)')

#7  How does body image impact scores differ by state and region?
plt.figure(figsize=(10, 6))
sns.violinplot(x='Urban/Rural', y='Daily SM Usage (hrs)', data=data_df)
plt.title('Social Media Usage by Urban vs. Rural Areas')
plt.xlabel('Urban/Rural')
plt.ylabel('Daily Social Media Usage (hrs)')

#8  What is the state-wise distribution of sleep quality impact from social media use? 
plt.figure(figsize=(10, 6))
sns.heatmap(data_df[['Self Confidence Impact (1-10)', 'Anxiety Levels (1-10)']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Self-Confidence Impact and Anxiety Level')

#9  Are there significant differences in social media addiction rates across different countries? 
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Self Confidence Impact (1-10)', y='Sleep Quality Impact (1-10)', data=data_df)
plt.title('Self-Confidence Impact vs Sleep Quality')
plt.xlabel('Self Confidence Impact (1-10)')
plt.ylabel('Sleep Quality Impact (1-10)')

#10 How does the frequency of social media use vary across people living in urban and rural areas? 
plt.figure(figsize=(10, 6))
sns.countplot(x='State', hue='Frequency of SM Use', data=data_df)
plt.title('Proportions of Daily vs Monthly Social Media Users by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.legend(title='Frequency of SM Use')

#11 What is the correlation between self-confidence impact and anxiety level? 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_df, x='Self Confidence Impact (1-10)', y='Anxiety Levels (1-10)')
plt.title('Correlation between Self-Confidence Impact and Anxiety Level')
plt.xlabel('Self Confidence Impact')
plt.ylabel('Anxiety Level')

#12 How does self-confidence level affect sleep quality? 
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_df, x='Self Confidence Impact (1-10)', y='Sleep Quality Impact (1-10)')
plt.title('Self-Confidence Level vs. Sleep Quality Impact')
plt.xlabel('Self Confidence Impact')
plt.ylabel('Sleep Quality Impact')
plt.xticks(rotation=45)

#13 Which states have the highest proportions of daily social media users compared to monthly users? 
user_counts = data_df.groupby(['State', 'Frequency of SM Use']).size().unstack().fillna(0)

user_proportions = user_counts.div(user_counts.sum(axis=1), axis=0)

user_proportions.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Proportion of Daily vs. Monthly Social Media Users by State')
plt.xlabel('State')
plt.ylabel('Proportion of Users')
plt.legend(title='Frequency of SM Use', loc='upper right')

#14 Which regions show the most diversity in terms of the number of social media platforms used? 
plt.figure(figsize=(12, 6))
sns.countplot(data=data_df, x='Most Used SM Platform', hue='Region')
plt.title('Diversity of Social Media Platforms Used by Region')
plt.xlabel('Most Used Social Media Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Region')

#15 Which states experience the highest frequency of cyberbullying on social media?
cyberbullying_by_state = data_df.groupby('State')['Cyberbullying Experience (1-10)'].mean().reset_index()

cyberbullying_by_state = cyberbullying_by_state.sort_values(by='Cyberbullying Experience (1-10)', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=cyberbullying_by_state, x='State', y='Cyberbullying Experience (1-10)', palette='viridis')
plt.title('Average Cyberbullying Experience by State')
plt.xlabel('State')
plt.ylabel('Average Cyberbullying Experience (1-10)')
plt.xticks(rotation=45)

#16 How do different states report the impact of social media usage on physical health, such as sleep deprivation or sedentary behavior? 
health_impact_by_state = data_df.groupby('State')['Physical Health Impact'].mean().reset_index()

health_impact_by_state = health_impact_by_state.sort_values(by='Physical Health Impact', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=health_impact_by_state, x='State', y='Physical Health Impact', palette='coolwarm')
plt.title('Average Physical Health Impact of Social Media Usage by State')
plt.xlabel('State')
plt.ylabel('Average Physical Health Impact Score')
plt.xticks(rotation=45)
plt.tight_layout() 

#17 How do anxiety levels vary by education level among social media users? 
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_df, x='Education Level', y='Anxiety Level', palette='Set2')
plt.title('Anxiety Levels by Education Level Among Social Media Users')
plt.xlabel('Education Level')
plt.ylabel('Anxiety Level (1-10)')
plt.xticks(rotation=45)
plt.tight_layout()

#18 How does socioeconomic status influence the likelihood of experiencing anxiety due to social media? 
#19 How does the frequency of social media use differ between urban and rural areas? 
#20 Is there a significant relationship between the number of like received on ports and anxiety levels? 

