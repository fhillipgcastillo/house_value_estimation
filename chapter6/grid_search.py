import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.model_selection import GridSearchCV


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

# print(X[0])

# Split the data set in a training and tests sets 70/30 %
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# create the model with not prameters Fit regression model using GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()

# Parameters we want to try
param_grid = {
  'n_estimators':[500,1000,3000],# how many decision to create to be more created
  'learning_rate':[0.1],
  'max_depth':[4,6], #layer deep
  'min_samples_leaf': [3,5,9,17],
  'max_features':[1.0, 0.3, 0.1],
  'loss':['ls', 'lad', 'huber']
}

# Define the grid search we want to run. Run it with 4 cpus in parallel
gs_cv = GridSearchCV(model, param_grid, n_jobs=4)

# RUn the grid search - on only the training data
gs_cv.fit(x_train, y_train)

# print the parameters tha gave us the best result
print(gs_cv.best_params_)

# after runing the output will look like
# {'loss': 'huber', learning_rate':0.1,''}...
#  n_estimators=1000,
    # learning_rate=0.1,
    # max_depth=6,
    # min_samples_leaf=9,
    # max_features=0.1,
    # loss='huber',
    # random_state=0

# Find the error rate on the training set
mse =  mean_absolute_error(y_train, model.predict(x_train))
print("Training Set mean absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse =  mean_absolute_error(y_test, model.predict(x_test))
print("Test Set mean absolute Error: %.4f" % mse)
