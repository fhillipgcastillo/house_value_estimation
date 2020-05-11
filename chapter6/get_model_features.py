import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib

def get_features_index():
  df = pd.read_csv("../chapter3/ml_house_data_set.csv")

  #remove the fields from the dataset that were not included in the model
  del df['house_number']
  del df['unit_number']
  del df['street_name']
  del df['zip_code']
  del df['sale_price']

  features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

  # Output: 
  # Index(['year_built', 'stories', 'num_bedrooms', 'full_bathrooms',
  #      'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft',
  #      'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating',
  #      'has_central_cooling', 'garage_type_attached', 'garage_type_detached',
  #      'garage_type_none', 'city_Amystad', 'city_Brownport', 'city_Chadstad',
  #      'city_Clarkberg', 'city_Coletown', 'city_Davidfort', 'city_Davidtown',
  #      'city_East Amychester', 'city_East Janiceville', 'city_East Justin',
  #      'city_East Lucas', 'city_Fosterberg', 'city_Hallfort',
  #      'city_Jeffreyhaven', 'city_Jenniferberg', 'city_Joshuafurt',
  #      'city_Julieberg', 'city_Justinport', 'city_Lake Carolyn',
  #      'city_Lake Christinaport', 'city_Lake Dariusborough', 'city_Lake Jack',
  #      'city_Lake Jennifer', 'city_Leahview', 'city_Lewishaven',
  #      'city_Martinezfort', 'city_Morrisport', 'city_New Michele',
  #      'city_New Robinton', 'city_North Erinville', 'city_Port Adamtown',
  #      'city_Port Andrealand', 'city_Port Daniel', 'city_Port Jonathanborough',
  #      'city_Richardport', 'city_Rickytown', 'city_Scottberg',
  #      'city_South Anthony', 'city_South Stevenfurt', 'city_Toddshire',
  #      'city_Wendybury', 'city_West Ann', 'city_West Brittanyview',
  #      'city_West Gerald', 'city_West Gregoryview', 'city_West Lydia',
  #      'city_West Terrence'],
  #     dtype='object')
  return features_df.keys()
