#!/bin/bash
# Debug logs for troubleshooting
echo "Starting application..." > /home/LogFiles/startup.log
echo "Current directory: $(pwd)" >> /home/LogFiles/startup.log
echo "Files in directory:" >> /home/LogFiles/startup.log
ls -la >> /home/LogFiles/startup.log

# Make sure we're in the correct directory
cd /home/site/wwwroot
echo "After cd, current directory: $(pwd)" >> /home/LogFiles/startup.log

# Set environment variables
export PYTHONUNBUFFERED=1

# Start gunicorn with a single worker for stability
gunicorn --bind=0.0.0.0:8000 --timeout=600 --workers=1 app:app >> /home/LogFiles/gunicorn.log 2>&1
