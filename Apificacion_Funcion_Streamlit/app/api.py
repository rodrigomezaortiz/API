from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import io
import numpy as np 

# Inicializar la aplicación FastAPI 
app = FastAPI()
# Cargar el modelo previamente entrenado
model_path = '/app/models/final_model.pkl' 
model = joblib.load(model_path, mmap_mode='r') 

# Definir las columnas esperadas en los datos de entrada
expected_columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday',
                    'Year', 'Month', 'WeekOfYear', 'IsHoliday', 'Sales_Lag1', 
                    'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']


# Ruta raíz para verificar que la API funciona 
@app.get("/")
def read_root():
    return {"message": "Welcome to the Rossmann API"}


# Endpoint para cargar los datos y obtener predicciones
@app.post("/predict")
async def predict_sales(file: UploadFile = File(...)):
    try:
        # Leer el archivo CSV subido
        contents = await file.read()
        test_data = pd.read_csv(io.BytesIO(contents))
        
        # Verificar si el archivo contiene todas las columnas necesarias
        missing_columns = [col for col in expected_columns if col not in test_data.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Faltan las siguientes columnas en los datos cargados: {missing_columns}")
        
        # Preprocesar los datos (misma lógica de tu código Streamlit)
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek
        test_data['WeekOfYear'] = test_data['Date'].dt.isocalendar().week
        test_data['IsHoliday'] = np.where(
            (test_data[['StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']].sum(axis=1) > 0) | 
            (test_data['SchoolHoliday'] == 1), 
            1, 
            0
        )
        
        if 'Sales' in test_data.columns:
            X_test = test_data.drop(columns=['Date', 'Sales'])
        else:
            X_test = test_data.drop(columns=['Date'])
        
        # Hacer las predicciones
        predictions = model.predict(X_test)
        test_data['Predicted Sales'] = predictions
        
        # Devolver las primeras 5 predicciones como ejemplo
        return test_data[['Store', 'Date', 'Predicted Sales']].head().to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {e}")

# Comando para correr la API en localhost en el puerto 8000 (lanzarla desde Docker o localmente con `fastapi`)












