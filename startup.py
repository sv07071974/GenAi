# startup.py
import app
import os
import sys

# Define the port
port = int(os.environ.get("PORT", 8000))

# Start the Gradio app
if __name__ == "__main__":
    app.demo.launch(server_name="0.0.0.0", server_port=port)
