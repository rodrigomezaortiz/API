import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, Ridge
from sklearn.cluster import KMeans
import joblib

train_data = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/processed/train_data_processed.csv')

X = train_data.drop(columns=['Sales', 'Date'])
y = train_data['Sales']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/train/X_train.csv', index=False)
X_val.to_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/test/X_val.csv', index=False)
y_train.to_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/train/y_train.csv', index=False)
y_val.to_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/test/y_val.csv', index=False)

models = {
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge Regression': Ridge(random_state=42),
    'Clustering': KMeans(n_clusters=4, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/models/trained_model_{name.replace(" ", "_").lower()}.pkl')

final_model = models['Random Forest']
joblib.dump(final_model, 'PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/models/final_model.pkl') 
