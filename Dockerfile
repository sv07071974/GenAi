# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster # Or your desired python version

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Specify the command to run on container startup
CMD ["python", "your_app.py"] # Replace your_app.py with your main python file.
