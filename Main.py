import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Social-Media-Health-Impact/Main/Mental_Health_Survey_Feb_20_22.csv')
for column in df.columns:
  missing_values = df[column].isnull().sum()
  print(f"Column '{column}' has {missing_values} missing value(s).")



