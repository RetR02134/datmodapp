import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('final_data.csv')  # Make sure to upload your dataset

feature_columns = [
    'RoundNumber', 'eventYear', 'Stint', 'meanAirTemp', 'meanTrackTemp', 
    'meanHumid', 'GridPosition', 'raceStintsNums', 'TyreAge', 
    'lapNumberAtBeginingOfStint', 'StintLen', 'CircuitLength', 'designedLaps', 
    'deg_slope', 'deg_bias'
]

target_column = 'bestPreRaceTime'
X = data[feature_columns]
y = data[target_column]

# Define a threshold for classifying bad pit stops
threshold = st.sidebar.slider('Define Threshold for Bad Pit Stop', min_value=float(y.min()), max_value=float(y.max()), value=float(y.mean()))

# Create a new column in the dataset for classification
data['is_bad_pit_stop'] = data[target_column] > threshold
y_class = data['is_bad_pit_stop'].astype(int)

# Split the data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

# Train and evaluate KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
param_grid_class = {'n_neighbors': list(range(1, 31))}
grid_search_class = GridSearchCV(knn_classifier, param_grid_class, cv=5, scoring='accuracy')
grid_search_class.fit(X_train_class_scaled, y_train_class)
optimal_k_class = grid_search_class.best_params_['n_neighbors']
st.write(f"Optimal k value for classification: {optimal_k_class}")

knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k_class)
knn_classifier.fit(X_train_class_scaled, y_train_class)

# Train and evaluate RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators=100, random_state=42)
rf_class.fit(X_train_class, y_train_class)

# User input for new pit stop
st.sidebar.header('Input New Pit Stop Data')
def user_input_features():
    data = {
        'RoundNumber': st.sidebar.number_input('Round Number', min_value=1),
        'eventYear': st.sidebar.number_input('Event Year', min_value=1950, max_value=2024),
        'Stint': st.sidebar.number_input('Stint', min_value=1),
        'meanAirTemp': st.sidebar.number_input('Mean Air Temperature (°C)'),
        'meanTrackTemp': st.sidebar.number_input('Mean Track Temperature (°C)'),
        'meanHumid': st.sidebar.number_input('Mean Humidity (%)'),
        'GridPosition': st.sidebar.number_input('Grid Position', min_value=1),
        'raceStintsNums': st.sidebar.number_input('Number of Race Stints', min_value=1),
        'TyreAge': st.sidebar.number_input('Tyre Age (laps)', min_value=1),
        'lapNumberAtBeginingOfStint': st.sidebar.number_input('Lap Number at Beginning of Stint', min_value=1),
        'StintLen': st.sidebar.number_input('Stint Length (laps)', min_value=1),
        'CircuitLength': st.sidebar.number_input('Circuit Length (km)', min_value=1.0),
        'designedLaps': st.sidebar.number_input('Designed Laps', min_value=1),
        'deg_slope': st.sidebar.number_input('Deg Slope'),
        'deg_bias': st.sidebar.number_input('Deg Bias')
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.write(input_df)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict using the trained classifiers
prediction_knn = knn_classifier.predict(input_scaled)
prediction_rf = rf_class.predict(input_scaled)

st.subheader('Prediction using KNN Classifier')
st.write('Bad Pit Stop' if prediction_knn[0] else 'Good Pit Stop')

st.subheader('Prediction using Random Forest Classifier')
st.write('Bad Pit Stop' if prediction_rf[0] else 'Good Pit Stop')

