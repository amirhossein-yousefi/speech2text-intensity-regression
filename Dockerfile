FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860


CMD ["python", "app/app.py"]
