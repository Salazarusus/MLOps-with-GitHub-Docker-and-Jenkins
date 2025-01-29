# Import packages
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Set path for the input
RAW_DATA_DIR = os.environ["RAW_DATA_DIR"]
RAW_DATA_FILE = os.environ["RAW_DATA_FILE"]
raw_data_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

# Read dataset
df = pd.read_csv(raw_data_path, sep=",")
 
# Encode variables 
label_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder(categories=[['unfurnished','semi-furnished','furnished']])
df['mainroad'] = label_encoder.fit_transform(df['mainroad'])
df['guestroom'] = label_encoder.fit_transform(df['guestroom'])
df['hotwaterheating'] = label_encoder.fit_transform(df['hotwaterheating'])
df['airconditioning'] = label_encoder.fit_transform(df['airconditioning'])
df['prefarea'] = label_encoder.fit_transform(df['prefarea'])
df['furnishingstatus'] = ordinal_encoder.fit_transform(df[['furnishingstatus']])
df['basement'] = label_encoder.fit_transform(df['basement'])

# Split into train and test
train, test = train_test_split(df, test_size=0.3)

# Set path to the outputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# Save csv
train.to_csv(train_path, index=False)
test.to_csv(test_path,  index=False)   