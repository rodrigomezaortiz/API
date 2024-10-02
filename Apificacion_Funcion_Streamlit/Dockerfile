FROM python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de tu aplicación al contenedor
COPY ./app /app
COPY ./models /app/models
COPY requirements.txt /app/

# Actualiza el sistema y instala 'distutils'
RUN apt-get update && apt-get install -y python3-distutils

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto de la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
