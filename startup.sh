echo '#!/bin/bash
cd /home/site/wwwroot
export PYTHONUNBUFFERED=1
python -m gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers=1 app:app' > startup.sh
