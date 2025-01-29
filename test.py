# Import packages
import pandas as pd
from joblib import load
import json
import os
from sklearn.metrics import r2_score 

# Set path for the input (model)
MODEL_DIR = os.environ["MODEL_DIR"]
model_file = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

# Set path for the input (test data)
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
test_data_file = 'test.csv'
test_data_path = os.path.join(PROCESSED_DATA_DIR, test_data_file)

# Load model
rf_model = load(model_path)

# Load data
df = pd.read_csv(test_data_path, sep=",")

# Split data into dependent and independent variables
X_test = df.drop('price', axis=1)
y_test = df['price']

# Predict
rf_predictions = rf_model.predict(X_test)

# Compute test accuracy
test_rf = r2_score(y_test, rf_predictions) 

# Test accuracy to JSON
test_metadata = {
    'test_r2': test_rf
}

# Set output path
RESULTS_DIR = os.environ["RESULTS_DIR"]
test_results_file = 'test_metadata.json'
results_path = os.path.join(RESULTS_DIR, test_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(test_metadata, outfile) 