FROM python:3.10-slim
WORKDIR /app

# 1. Copy the wheels folder and requirements file into the container
COPY wheels /app/wheels
COPY requirements.txt .

# 2. Force pip to install requirements AND xgboost from the local wheels folder
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt xgboost

# 3. Copy the rest of your project files
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
