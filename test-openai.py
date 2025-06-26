#!/usr/bin/env python3
"""
Test OpenAI client initialization to debug proxy issues
"""

import os
import sys

# Clear all proxy environment variables
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
cleared_vars = []
for var in proxy_vars:
    if var in os.environ:
        cleared_vars.append(f"{var}={os.environ[var]}")
        del os.environ[var]

if cleared_vars:
    print("Cleared proxy variables:", cleared_vars)
else:
    print("No proxy variables found")

# Check for OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

print("OpenAI API key found")

# Test OpenAI client initialization
try:
    from openai import OpenAI
    print("OpenAI library imported successfully")
    
    print("Creating OpenAI client...")
    client = OpenAI(
        api_key=api_key,
        timeout=60.0,
        max_retries=3
    )
    print("OpenAI client created successfully!")
    
    # Test a simple API call
    print("Testing API call...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, just testing the connection."}],
        max_tokens=10
    )
    print("API call successful!")
    print("Response:", response.choices[0].message.content)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()