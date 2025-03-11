#!/bin/bash
export PYTHONUNBUFFERED=1
echo "Starting application..." > /home/LogFiles/startup.log 2>&1
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers=1 app:app >> /home/LogFiles/gunicorn.log 2>&1
