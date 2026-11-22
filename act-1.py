import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('heart.csv')
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.isnull().sum())
df.info()

plt.figure(figsize=(8,5))
plt.hist(df['age'], bins=10, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of people')
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['chol'])
plt.title('cholesterol distribution')
plt.xlabel('Cholesterol level')
plt.ylabel('Number of people')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x=df['sex'], y=df['chol'], hue=df['target'], data=df)
plt.title('Average Cholesterol by Sex')
plt.xlabel('Sex')
plt.ylabel('Average Cholesterol Level')
plt.show()

print(df['sex'].value_counts())
print(df['target'].value_counts())
print(df['thal'].value_counts())

plt.figure(figsize=(12,8))
correlation=df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='ca', hue='target', data=df)
plt.xlabel('ca (number of major vessels colored by fluoroscopy)')
plt.ylabel('Number of People')
plt.title('Distribution of ca by Target')
plt.show()