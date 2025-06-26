#!/usr/bin/env python3
"""
Local development server for Medical Records Summary Tool
Run this on Windows without Google Cloud dependencies
"""

import os
import sys

# Set local mode
os.environ['LOCAL_MODE'] = 'true'

# Check for OpenAI API key
if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: Please set OPENAI_API_KEY environment variable")
    print("Example (Windows): set OPENAI_API_KEY=your-api-key-here")
    print("Example (PowerShell): $env:OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

# Import and run the Flask app
from server import app

if __name__ == '__main__':
    print("Starting Medical Records Summary Tool in LOCAL MODE...")
    print("Server will run on http://localhost:5000")
    print("Make sure you have set OPENAI_API_KEY environment variable")
    print("Files will be stored in ./local_uploads directory")
    
    # Run the development server
    app.run(debug=True, host='0.0.0.0', port=5000)