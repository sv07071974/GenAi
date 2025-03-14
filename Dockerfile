# Use an official Python image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports (if applicable, e.g., for a web app)
EXPOSE 5000

# Specify the command to run on container startup
CMD ["python", "wsgi.py"] # Replace your_app.py with your main python file
