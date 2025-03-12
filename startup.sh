# New startup.sh
#!/bin/bash
cd /home/site/wwwroot
cat > mini_app.py << 'EOF'
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'Hello World!']
EOF
gunicorn --bind=0.0.0.0:8000 mini_app:app
