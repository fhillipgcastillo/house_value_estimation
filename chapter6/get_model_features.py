import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("../chapter3/ml_house_data_set.csv")

#remove the fields from the dataset that wont want to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

print('features key')
print(df.keys())
