#!/bin/bash

# Start Gunicorn with appropriate settings for Azure
gunicorn --bind=0.0.0.0:$PORT --timeout 600 --workers 2 --threads 2 app:app
