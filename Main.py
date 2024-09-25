import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#data_df = pd.read_csv('/Users/pavlovovk/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv')
#data_df = pd.read_csv('/Users/gopivaghani/Documents)
#data_df = pd.read_csv('/Users/amaankhan/Documents/GitHub/Social-Media-Health-Impact/South_East_Asia_Social_Media_MentalHealth.csv')

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

#11 What is the correlation between socioeconomic status and social anxiety levels?

#12 Does daily social media usage vary significantly between different education levels?

#13 How do peer comparison frequencies differ across countries?

#14 What is the relationship between the number of likes received and social anxiety levels?

#15 How does body image impact scores vary across different age groups and genders?

#16 Is there a significant difference in sleep quality impact between users who frequently experience cyberbullying and those who don’t?

#17 How does the frequency of social media usage impact self-confidence across different socioeconomic statuses?

#18 Which country has the highest body image impact score?

#19 How does the number of comments received on posts affect social anxiety levels?

#20 What is the relationship between anxiety levels and the number of shares received per post?


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
sns.scatterplot(x='Daily SM Usage (hrs)', y='Social Anxiety Level (1-10)', hue='Custom Cluster', data=data_df, palette='viridis')
plt.title(f'Custom KMeans Clustering (k={k})')
plt.ylim(0, 15)
plt.show()

# Compare with Sklearn KMeans
sklearn_kmeans = KMeans(n_clusters=k, random_state=42)
sklearn_kmeans.fit(df_scaled)
sklearn_clusters = sklearn_kmeans.predict(df_scaled)
data_df['Sklearn Cluster'] = sklearn_clusters

# Display sklearn SSE
sklearn_sse = sklearn_kmeans.inertia_
print(f"Sklearn KMeans SSE for k={k}: {sklearn_sse}")

# Additional Visualization and Analysis
sns.scatterplot(x='Daily SM Usage (hrs)', y='Social Anxiety Level (1-10)', hue='Sklearn Cluster', data=data_df, palette='coolwarm')
plt.title(f'Sklearn KMeans Clustering (k={k})')
plt.ylim(0, 15)
plt.show()
