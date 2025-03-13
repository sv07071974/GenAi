#!/bin/bash

# Start Gunicorn with appropriate settings for Azure
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 --threads 8 app:app
