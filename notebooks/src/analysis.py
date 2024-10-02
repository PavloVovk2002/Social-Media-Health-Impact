import seaborn as sns
import matplotlib.pyplot as plt

def plot_social_media_usage(df):
    #Plot daily social media usage across age groups.
    sns.boxplot(x='Age Group', y='Daily SM Usage (hrs)', data=df)
    plt.title('Daily SM Usage Across Age Groups')
    plt.show()
