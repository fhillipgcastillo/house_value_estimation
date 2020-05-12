import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib

def clean_features(df, columns=[]):
  for col in columns:
    del df[col]

def get_xy(features, origin_df, res_column):
  # returns X, Y numpy arrays
  return features.to_numpy(), origin_df[res_column].to_numpy()

def show_mean_absolute_erros(model, _y, _x):
  mse = mean_absolute_error(_y, model.predict(_x))
  print("Training Set Mean Absolute Error: %.4f" % mse)
  
def get_train_test_splitted(features_df, df, res_column):
  # Create the X and y arrays
  X, y = get_xy(features_df, df, res_column)
# Split the data set in a training set (70%) and a test set (30%)
  return train_test_split(X, y, test_size=0.3, random_state=0)

def train_model(model, X_train, y_train):
  print("Training...")
  model.fit(X_train[0:100], y_train[0:100])
  print ("Model trained")
  return model

def create_model():
  return ensemble.GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
  )

def get_trained_model( X_train, y_train ):
  return train_model(create_model(), X_train, y_train)

# Step 1: get data
# Load the data set
df = pd.read_csv("ml_house_data_set.csv")

# Step 2: clean data for ml
# Remove the fields from the data set that we don't want to include in our model
clean_features(df, ['house_number','unit_number', 'street_name','zip_code'])

# Step 3: use on-hot encoded data 
# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Step 4: remove result column from features
# Remove the sale price from the feature data
# Reuse existent procedure/function
res_column = 'sale_price'
clean_features(features_df, [res_column])

# Step 4: get x,y train and test data
X_train, X_test, y_train, y_test =  get_train_test_splitted(features_df, df, res_column)

# Fit regression model with train data
model = get_trained_model(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
# joblib.dump(model, 'trained_house_classifier_model.pkl')

# Find the error rate on the training set
show_mean_absolute_erros(model, y_train, X_train)
# Find the error rate on the test set
show_mean_absolute_erros(model, y_test, X_test)

