FROM python:3.10

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8501

# CMD ["streamlit", "run", "main.py"]
CMD ["uvicorn", "api_app:api_app", "--host", "0.0.0.0", "--port", "8501"]