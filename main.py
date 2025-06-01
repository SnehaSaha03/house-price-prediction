import pandas as pd

# Read the CSV files
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Print the first 5 rows
print(train_data.head())
