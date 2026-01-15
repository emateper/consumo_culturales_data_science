FROM python:3.11-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación completa
COPY . .

# Exponemos el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
