# Local Development Setup (Windows)

This branch allows you to run the Medical Records Summary Tool locally on Windows without Google Cloud dependencies.

## Prerequisites

1. **Python 3.9+** installed
2. **OpenAI API Key** (get from https://platform.openai.com/api-keys)
3. **No C++ build tools required** - uses pure Python dependencies

## Setup Instructions

1. **Clone and switch to local branch**:
   ```bash
   git clone <your-repo-url>
   cd MedGPT
   git checkout local-development
   ```

2. **Install dependencies** (pure Python, no C++ compiler needed):
   ```bash
   pip install -r requirements-local.txt
   ```

3. **Set your OpenAI API Key**:
   
   **Command Prompt:**
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```
   
   **PowerShell:**
   ```powershell
   $env:OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run the application**:
   ```bash
   python run-local.py
   ```

5. **Open your browser** and go to: http://localhost:5000

## How It Works

- Files are uploaded to `./local_uploads` directory instead of Google Cloud Storage
- No Google Cloud dependencies required
- Same functionality as the cloud version
- Automatic cleanup of temporary files after processing

## Features Available

- ✅ Upload medical records (PDFs, images, etc.)
- ✅ AI-powered medical record analysis
- ✅ Generate medical overview documents
- ✅ Generate medical chronology documents
- ✅ Large file support (no 32MB Cloud Run limit)
- ✅ Progress tracking with real-time updates

## Troubleshooting

**"OpenAI API key not found"**: Make sure you've set the environment variable correctly

**"Permission denied"**: Run as administrator or check file permissions

**"Module not found"**: Make sure you installed requirements-local.txt

**PDF processing issues**: The app uses pypdf (pure Python) instead of PyMuPDF for better Windows compatibility

## File Structure

```
local_uploads/          # Uploaded files (created automatically)
uploads/               # Temporary processing files
requirements-local.txt # Dependencies without Google Cloud
run-local.py          # Local development server
README-local.md       # This file
```