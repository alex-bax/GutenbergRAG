# Base image with Python
FROM python:3.12-slim

# Set working directory
WORKDIR /mobyRag

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code over
COPY . .

# Expose FastAPI port 
EXPOSE 8080

# Run the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]