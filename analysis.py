import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import optuna
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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

# Drop columns that are completely NaN or have insufficient non-missing values for imputation
columns_to_drop = ['Client_Occupation', 'Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Data Visualization: Distribution of Loan Default
plt.figure(figsize=(8, 6))
sns.countplot(x='Default', data=df)
plt.title('Distribution of Loan Default (1 = Default, 0 = No Default)')
plt.xlabel('Default')
plt.ylabel('Count')
plt.savefig('graphs/loan_default_distribution.png')  # Save the plot

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
    plt.savefig(f'graphs/distribution_of_{feature}.png')  # Save each plot individually

plt.tight_layout()
plt.show()

# Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('graphs/correlation_heatmap.png')  # Save the plot

plt.show()


# Create a pipeline for preprocessing with an imputer for numeric columns
numeric_transformer = SimpleImputer(strategy='median')
preprocessor = Pipeline(steps=[('imputer', numeric_transformer)])

# Splitting features and target variable
X = df.drop(columns=['Default'])
y = df['Default']

# Preprocess the data using the pipeline
X_processed = preprocessor.fit_transform(X)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Create an Optuna study
study = optuna.create_study(study_name='my_study',  storage="sqlite:///db.sqlite3", load_if_exists=True)

# Define an objective function for Optuna
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return classification_report(y_test, preds, output_dict=True)['1']['f1-score']

# Run Optuna optimization
study.optimize(objective, n_trials=1)

# Train the final model with best parameters
best_params = study.best_params
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
# Model Evaluation: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('graphs/confusion_matrix.png')  # Save the confusion matrix as a graph

plt.show()


# Model Evaluation: Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:\n", report)

# Create a heatmap of the classification report
plt.imshow([[report['accuracy'], report['macro avg']['f1-score']], 
            [report['weighted avg']['precision'], report['weighted avg']['recall']]], 
           cmap='Blues', interpolation='nearest')
plt.title('Classification Report')
plt.xlabel('Metrics')
plt.ylabel('Average')
plt.xticks([0, 1], ['Accuracy', 'Macro F1'])
plt.yticks([0, 1], ['Weighted Precision', 'Weighted Recall'])
plt.colorbar()
plt.savefig('graphs/classification_report.png')  # Save the classification report as a graph

plt.show()

# Get feature importances
feature_importances = model.feature_importances_

# Sort features by importance in descending order
sorted_importances = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)

# Plot feature importances
plt.bar([x[0] for x in sorted_importances], [x[1] for x in sorted_importances])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.savefig('graphs/feature_importance.png')
