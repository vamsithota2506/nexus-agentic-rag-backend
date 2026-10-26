import sys
import os

# Add root repository directory to sys.path so 'app' imports resolve
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.main import app

# Handler for Vercel serverless function
# Vercel ASGI adapter automatically binds to 'app'
