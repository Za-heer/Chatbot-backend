FROM python:3.10-slim

# System deps (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Start the API
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
