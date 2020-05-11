import numpy as np
import joblib
from get_model_features import get_features_index

# These are the feature labes from our data set
# features_labels = np.array(['year_built', 'stories', 'num_bedrooms', 'full_bathrooms', 'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft', 'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating', 'has_central_cooling', 'garage_type_attached', 'garage_type_detached', 'garage_type_none', 'city_Amystad', 'city_Brownport', 'city_Chadstad', 'city_Clarkberg', 'city_Coletown', 'city_Davidfort', 'city_Davidtown', 'city_East Amychester', 'city_East Janiceville', 'city_East Justin', 'city_East Lucas', 'city_Fosterberg', 'city_Hallfort', 'city_Jeffreyhaven', 'city_Jenniferberg', 'city_Joshuafurt', 'city_Julieberg', 'city_Justinport', 'city_Lake Carolyn', 'city_Lake Christinaport', 'city_Lake Dariusborough', 'city_Lake Jack', 'city_Lake Jennifer', 'city_Leahview', 'city_Lewishaven', 'city_Martinezfort', 'city_Morrisport', 'city_New Michele', 'city_New Robinton', 'city_North Erinville', 'city_Port Adamtown', 'city_Port Andrealand', 'city_Port Daniel', 'city_Port Jonathanborough', 'city_Richardport', 'city_Rickytown', 'city_Scottberg', 'city_South Anthony', 'city_South Stevenfurt', 'city_Toddshire', 'city_Wendybury', 'city_West Ann', 'city_West Brittanyview', 'city_West Gerald', 'city_West Gregoryview', 'city_West Lydia', 'city_West Terrence'])
features_labels = get_features_index()

# load the trained model created with train model.py
model = joblib.load(r"trained_house_classifier_model.pkl")

# create a numpy array based on the models feature importances
importance = model.feature_importances_

print(dir(importance))

# sort the features labels base on the feature importance rankis fom the model
features_inx_by_importance = importance.argsort()

# print each feature label, from importance to least mportance (reverse order)
for index in features_inx_by_importance:
  print("{} - {:.2f}%".format(features_labels[index], (importance[index] * 100.0)))

