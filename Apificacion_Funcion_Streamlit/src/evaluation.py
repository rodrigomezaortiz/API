import pandas as pd
import numpy as np
import joblib

X_val = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/test/X_val.csv')
y_val = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/test/y_val.csv')

model = joblib.load('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/models/final_model.pkl')

predictions = model.predict(X_val)
rmspe = np.sqrt(np.mean(((y_val - predictions) / y_val) ** 2))
print(f'Random Forest RMSPE: {rmspe:.4f}')

with open('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/models/model_evaluation.txt', 'w') as file:
    file.write(f'Random Forest RMSPE: {rmspe:.4f}\n') 
