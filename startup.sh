#!/bin/bash
# Debug logs
echo "Starting application..." > /home/LogFiles/startup.log
echo "Current directory: $(pwd)" >> /home/LogFiles/startup.log
echo "Files in directory:" >> /home/LogFiles/startup.log
ls -la >> /home/LogFiles/startup.log

# Make sure we're in the correct directory
cd /home/site/wwwroot
echo "After cd, current directory: $(pwd)" >> /home/LogFiles/startup.log

# Activate the Python virtual environment
source /home/site/wwwroot/antenv/bin/activate
echo "Python path: $(which python)" >> /home/LogFiles/startup.log
echo "Python version: $(python --version)" >> /home/LogFiles/startup.log

# Install requirements
pip install -r requirements.txt >> /home/LogFiles/pip_install.log 2>&1

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/site/wwwroot

# Start gunicorn with a single worker
python -m gunicorn --bind=0.0.0.0:8000 --timeout=600 --workers=1 app:app >> /home/LogFiles/gunicorn.log 2>&1
