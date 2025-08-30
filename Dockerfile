# Use the official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files into the image
COPY . .

# Expose the port uvicorn will run on
EXPOSE 7860

# Run the app using uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]