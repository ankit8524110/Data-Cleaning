import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_path = "Titanic-Dataset.csv"
df = pd.read_csv(file_path)

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop(columns=['Cabin'])

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df = df.drop(columns=['Name', 'Ticket', 'PassengerId'])

scaler = StandardScaler()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols = [col for col in num_cols if col != 'Survived']

df[num_cols] = scaler.fit_transform(df[num_cols])

plt.boxplot(df['Fare'])
plt.title("Boxplot of Fare After Scaling")
plt.show()

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

print("\nFinal Dataset Preview:\n")
print(df.head())

df.to_csv("titanic_cleaned.csv", index=False)

print("\n✅ Data Cleaning & Preprocessing Completed Successfully!")