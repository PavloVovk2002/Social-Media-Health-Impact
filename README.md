Social Media & Mental Health Impact in Southeast Asia

Overview
This project explores the impact of social media on mental health across different demographics in Southeast Asia. The dataset used contains variables such as daily social media usage, social anxiety, self-confidence impact, and cyberbullying experiences.


Data Cleaning and Preprocessing
Missing Values: We first identified the number of missing values per column and handled them using imputation or by removing rows with missing critical data.
Standardization: Variables like social media usage and mental health impact scores were standardized using `StandardScaler` to ensure consistent scaling for clustering analysis.
  

Analysis and Visualizations
1. Daily Social Media Usage Across Age Groups: A boxplot was created to show the distribution of daily usage across various age groups.
2. Most Used Platform by Gender: A count plot illustrated the most popular social media platforms across genders.
3. Urban vs. Rural Social Media Usage: A comparison between social media usage in urban and rural settings using boxplots.
4. Country-wise Social Anxiety Levels: A bar plot highlighted the average social anxiety level across different countries.
5. Peer Comparison and Social Anxiety: Scatterplots explored the relationship between how often people compare themselves to others and their social anxiety levels.
6. Cyberbullying by Country: We compared cyberbullying experiences across countries using boxplots.
7. Self-Confidence by Age Group: A bar plot showing the average self-confidence impact score across age groups.
8. Body Image Impact by Gender and Age Group: A detailed boxplot that visualizes body image concerns across gender and age.


Interesting Findings
Social Anxiety and Social Media Usage: Countries with higher daily social media usage also exhibited higher social anxiety levels, which may indicate a correlation between excessive usage and anxiety.
Peer Comparison and Self-Confidence: There was a notable negative correlation between frequent peer comparison and self-confidence, suggesting that comparing oneself to others on social media leads to a decline in self-esteem.
Cyberbullying Prevalence: Certain countries reported significantly higher cyberbullying experiences, which could be due to regional cultural differences or varying levels of social media penetration.


Future Analysis Questions
1. How do specific platforms (e.g., Instagram, Facebook) contribute differently to mental health outcomes like social anxiety or self-confidence?
2. Are there particular age groups more vulnerable to the negative impact of social media on body image and anxiety?
3. What role does socioeconomic status play in mediating the mental health impact of social media use?
4. Can we predict mental health outcomes (e.g., high anxiety levels) using a combination of social media usage patterns and demographic data?
5. How do sleep quality and social media usage interact across different genders and age groups?


Clustering
We applied both custom and `sklearn` KMeans clustering to segment the data based on variables like:
- Daily Social Media Usage
- Peer Comparison Frequency
- Social Anxiety Levels
- Self-Confidence Impact