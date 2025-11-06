import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("heart_disease.csv")

# Basic info
print("Shape of dataset:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop missing values for analysis
df = df.dropna()

# Correlation heatmap for numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution plots
for col in numeric_df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Class imbalance
target_col = 'Heart Disease Status'
sns.countplot(data=df, x=target_col, palette='Set2')
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (No/Yes)")
plt.ylabel("Number of Patients")
plt.show()

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    if col != target_col:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index[:5], hue=target_col, palette="pastel")
        plt.title(f"{col} Distribution by Heart Disease")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()



# Violin plots for deeper distribution comparison
for col in numeric_df.columns:
    if col != target_col:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=target_col, y=col, data=df, palette='pastel')
        plt.title(f"{col} Distribution by Heart Disease Status")
        plt.tight_layout()
        plt.show()

# Pairplot (limited to first few numerical features to avoid clutter)
selected_cols = list(numeric_df.columns[:4]) + [target_col]
sns.pairplot(df[selected_cols], hue=target_col, palette='husl', diag_kind='kde')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Skewness of features
print("\nSkewness of Numeric Features:")
print(numeric_df.skew().sort_values(ascending=False))

# Target correlation bar chart
cor_target = numeric_df.corr()[target_col].drop(target_col)
cor_target.sort_values().plot(kind='barh', color='teal', figsize=(8, 5))
plt.title("Correlation of Features with Heart Disease")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

# Simple Scatter Plot: Age vs Cholesterol
plt.figure(figsize=(7, 5))
plt.scatter(df['Age'], df['Cholesterol'], color='coral', alpha=0.6)
plt.title('Scatter Plot: Age vs Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# Simple Line Chart: Age vs RestingBP
plt.figure(figsize=(7, 5))
plt.plot(df['Age'], df['RestingBP'], color='green', linewidth=1.5)
plt.title('Line Chart: Age vs RestingBP')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
