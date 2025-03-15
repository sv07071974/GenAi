FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variable for debugging
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
