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
#   data_df.groupby('Age Group')['Anxiety Levels (1-10)'].mean().plot(kind='bar')
#   plt.title('Average Anxiety Levels by Age Group')

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
#   sns.boxplot(x='Cyberbullying Experience', y='Sleep Quality Impact (1-10)', data=data_df)
#   plt.title('Sleep Quality Impact by Cyberbullying Experience')

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

#30 What is the relationship between anxiety levels and the number of shares received per post?
#   sns.scatterplot(x='Shares Received (per post)', y='Anxiety Levels (1-10)', data=data_df)
#   plt.title('Shares Received vs Anxiety Levels')

# Cluster

columns_to_cluster = ['Daily SM Usage (hrs)', 'Peer Comparison Frequency (1-10)', 
                      'Social Anxiety Level (1-10)', 'Self Confidence Impact (1-10)']

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(data_df[columns_to_cluster]), columns=columns_to_cluster)
class CustomKMeans:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.cluster_assignments = None

    def initialize_centroids(self, df):
        return df.sample(n=self.k).values

    def assign_clusters(self, df):
        distances = np.linalg.norm(df.values[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def compute_centroids(self, df):
        centroids = np.zeros((self.k, df.shape[1]))
        for i in range(self.k):
            cluster_points = df[self.cluster_assignments == i]
            centroids[i] = cluster_points.mean(axis=0)
        return centroids

    def fit(self, df):
        self.centroids = self.initialize_centroids(df)
        for i in range(self.max_iters):
            self.cluster_assignments = self.assign_clusters(df)
            new_centroids = self.compute_centroids(df)
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
            self.centroids = new_centroids

    def predict(self, df):
        return self.assign_clusters(df)

    def sse(self, df):
        total_sse = 0
        for i in range(self.k):
            cluster_points = df[self.cluster_assignments == i]
            total_sse += np.sum((cluster_points - self.centroids[i]) ** 2)
        return total_sse

# Custom KMeans
k = 3
custom_kmeans = CustomKMeans(k=k)
custom_kmeans.fit(df_scaled)
predicted_clusters = custom_kmeans.predict(df_scaled)

# Predicted clusters
data_df['Custom Cluster'] = predicted_clusters

# Display SSE
sse_value = custom_kmeans.sse(df_scaled)
print(f"Custom KMeans SSE for k={k}: {sse_value}")

# Results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Daily SM Usage vs Social Anxiety
sns.scatterplot(ax=axes[0], x='Daily SM Usage (hrs)', y='Social Anxiety Level (1-10)', hue='Custom Cluster', data=data_df, palette='viridis')
axes[0].set_title('Daily SM Usage vs Social Anxiety')
axes[0].set_ylim(0, 14) 

# Plot 2: Peer Comparison Frequency vs Self Confidence Impact
sns.scatterplot(ax=axes[1], x='Peer Comparison Frequency (1-10)', y='Self Confidence Impact (1-10)', hue='Custom Cluster', data=data_df, palette='viridis')
axes[1].set_title('Peer Comparison vs Self Confidence')
axes[1].set_ylim(0,14)
plt.tight_layout()
plt.show()


# Compare with Sklearn KMeans
sklearn_kmeans = KMeans(n_clusters=k, random_state=42)
sklearn_kmeans.fit(df_scaled)
sklearn_clusters = sklearn_kmeans.predict(df_scaled)
data_df['Sklearn Cluster'] = sklearn_clusters

# Display sklearn SSE
sklearn_sse = sklearn_kmeans.inertia_
print(f"Sklearn KMeans SSE for k={k}: {sklearn_sse}")
