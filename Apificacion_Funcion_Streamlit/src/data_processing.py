import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

store_data = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/data/raw/store.csv')
train_data = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/raw/train.csv')
test_data = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/raw/test.csv')

store_data.fillna(method='ffill', inplace=True)

train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

train_data['Open'].fillna(1, inplace=True)

train_data['Year'] = train_data['Date'].dt.year
train_data['Month'] = train_data['Date'].dt.month
train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
train_data['WeekOfYear'] = train_data['Date'].dt.isocalendar().week
train_data['IsHoliday'] = np.where((train_data['StateHoliday'] != '0') | (train_data['SchoolHoliday'] == 1), 1, 0)

train_data = train_data.sort_values(by=['Store', 'Date'])
train_data['Sales_Lag1'] = train_data.groupby('Store')['Sales'].shift(1)
train_data['Sales_Lag1'].fillna(0, inplace=True)

train_data = pd.get_dummies(train_data, columns=['StateHoliday'], drop_first=True)
features_to_scale = ['Customers', 'Sales_Lag1']
scaler = StandardScaler()
train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])

store_data.to_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/processed/store_data_processed.csv', index=False)
train_data.to_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/data/processed/train_data_processed.csv', index=False)
test_data.to_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN/data/processed/test_data_processed.csv', index=False) 
