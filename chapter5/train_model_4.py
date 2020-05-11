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

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price from the feature data
del features_df['sale_price']

# Create the X and Y arrays
X = features_df.to_numpy()
Y = df['sale_price'].to_numpy()
# print(features_df.to_numpy())
# Split the data set in a training and tests sets 70/30 %
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=30/100)

# Fit regression model using GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor(
  n_estimators=1000,# how many decision to create to be more created
  learning_rate=0.1,
  max_depth=6, #layer deep
  min_samples_leaf=9,
  max_features=0.1,
  loss='huber'
)

model.fit(x_train, y_train)

# Save the trained model to a file so we can use it in other programs in future
joblib.dump(model, 'trained_house_classifier_model.pkl')

# Find the erro rate on the training set
mse =  mean_absolute_error(y_train, model.predict(x_train))
print("Training Set mean absolute Error: %.4f" % mse)

# Find the error rato on the test set
mse =  mean_absolute_error(y_test, model.predict(x_test))
print("Test Set mean absolute Error: %.4f" % mse)
