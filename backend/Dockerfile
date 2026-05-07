FROM python:3.11-slim

WORKDIR /app

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalacja pakietów Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj cały kod
COPY . .

EXPOSE 8000

# Poprawiona komenda startowa
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]