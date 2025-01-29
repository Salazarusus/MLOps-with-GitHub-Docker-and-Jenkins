# Import packages
import pandas as pd
import json
import os
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

# Set path to inputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_file = 'train.csv'
train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

# Read data
df = pd.read_csv(train_data_path, sep=",")

# Split data into dependent and independent variables
X_train = df.drop('price', axis=1)
y_train = df['price'] 

# Initialize RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Cross-validation
cv = KFold(n_splits=3)  # KFold for regression tasks
val_rf = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='r2').mean()  # Using R² score

# Validation score to JSON
train_metadata = {
    'validation_r2': val_rf
}

# Set path to output (model)
MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_name)

# Serialize and save model
dump(rf_model, model_path)

# Set path to output (metadata)
RESULTS_DIR = os.environ["RESULTS_DIR"]
train_results_file = 'train_metadata.json'
results_path = os.path.join(RESULTS_DIR, train_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(train_metadata, outfile) 