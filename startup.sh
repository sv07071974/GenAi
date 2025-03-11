#!/bin/bash
# Set environment variables for better debugging
export PYTHONUNBUFFERED=1

# Start gunicorn with debug logging
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers=1 --log-level debug app:app
