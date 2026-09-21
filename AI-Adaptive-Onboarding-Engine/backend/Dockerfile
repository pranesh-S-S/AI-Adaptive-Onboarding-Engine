FROM python:3.10-slim

# Install system dependencies required by pytesseract and pdf2image
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    poppler-utils \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
