import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with low_memory=False to avoid dtype warnings
df = pd.read_csv('data/Dataset.csv', low_memory=False)

# Remove dollar signs and convert to float using pd.to_numeric with error handling
numeric_columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].replace('[\$,]', '', regex=True), errors='coerce')

# Identify and clean any non-numeric data in numeric columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert categorical variables to numeric using one-hot encoding
categorical_columns = [
    'Accompany_Client', 'Client_Income_Type', 'Client_Education', 
    'Client_Marital_Status', 'Client_Gender', 'Loan_Contract_Type', 
    'Client_Housing_Type', 'Type_Organization'
]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Check for missing values and fill them for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Data Visualization: Distribution of Loan Default
plt.figure(figsize=(8, 6))
sns.countplot(x='Default', data=df)
plt.title('Distribution of Loan Default (1 = Default, 0 = No Default)')
plt.xlabel('Default')
plt.ylabel('Count')
plt.show()

# Advanced Analysis: Distribution of key features
features_to_plot = [
    'Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 
    'Child_Count', 'Own_House_Age', 'Mobile_Tag', 
    'Homephone_Tag', 'Workphone_Working', 'Credit_Bureau'
]

plt.figure(figsize=(20, 15))
for i, feature in enumerate(features_to_plot):
    plt.subplot(4, 3, i + 1)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Splitting features and target variable
X = df.drop(columns=['Default'])
y = df['Default']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Visualization
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()