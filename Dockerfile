# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to /app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports (if applicable, e.g., for a web app)
EXPOSE 7860

# Specify the command to run on container startup
CMD ["python", "app.py"]
