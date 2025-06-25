# Medical Records Summary Tool

A professional AI-powered tool for analyzing medical records and generating structured summaries for personal injury attorneys.

## Features

- **AI-Powered Analysis**: Uses OpenAI GPT-4.1-nano for medical record analysis
- **PDF Focus**: Optimized for PDF medical records with OCR support
- **OCR Support**: Extracts text from scanned documents and images
- **Structured Output**: Generates JSON with detailed medical record extraction
- **Lawyer Documents**: Creates professional Word documents for legal use
- **Auto-Split**: Splits large results to stay under size/word limits
- **Web Interface**: Professional web UI for easy use

## Generated Documents

1. **JSON Summary**: Structured data with individual medical records
2. **AI Medical Records Overview**: Professional Word document with:
   - High-level AI summary for legal strategy
   - Detailed breakdown by source file
   - Relevant medical records with exact provider quotes
3. **AI Medical Chronology**: Chronological table with all records sorted by date

## Installation

1. **Clone or download** this repository
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Tesseract OCR** (for scanned document processing):
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Web Interface (Recommended)

1. **Start the server**:
   ```bash
   python server.py
   ```

2. **Open browser** to http://localhost:5000

3. **Fill out the form**:
   - Select medical records folder
   - Enter client name
   - Describe the case and injury focus
   - Choose processing options

4. **Process and download** results

### Command Line

```bash
python ingest.py /path/to/medical/records \
  --client-name "Smith, John" \
  --case-prompt "Motor vehicle accident 01/15/2023. Focus on lower back injuries, herniated disc L4-L5." \
  --generate-lawyer-docs \
  --auto-split
```

## Medical Record Structure

The tool extracts individual medical records with these fields:

- **record_type**: primary_care_visit, emergency_room_visit, physical_therapy_visit, specialist_visit, other
- **date_of_visit**: MM/DD/YYYY format
- **notes_from_visit**: Exact quotes from medical notes
- **medication_treatment_changes**: Changes in medication or treatment
- **injury_condition_description**: Description of injury/condition/recovery
- **mentions_target_injury**: Boolean - relevant to the case (incident, target injuries, or related conditions)
- **pages_found_on**: Page numbers where record appears
- **record_summary**: Summary of the specific record

## Document-Level Fields

- **dates_covered**: Date range of the document
- **hospital_facility_provider**: Name of care provider
- **page_count**: Number of pages
- **overall_summary**: Summary of entire document

## File Size Limits

- **75,000 words** per JSON file
- **30MB** per JSON file
- Automatically splits larger results into multiple files with index

## Supported File Types

- **PDF**: Primary focus - text extraction with OCR fallback for scanned pages
- **Word Documents**: .docx and .doc files (supplementary)
- **Text Files**: .txt and .md files (supplementary)

## Requirements

- Python 3.8+
- OpenAI API key
- Tesseract OCR (for image/scanned document processing)
- 2GB+ RAM recommended for large document sets
- **50GB upload limit** - handles massive medical record collections

## Cost Estimation

Uses OpenAI GPT-4.1-nano ($0.10 per 1M tokens). Typical costs:
- Small case (10-20 documents): $0.01-$0.05
- Medium case (50-100 documents): $0.05-$0.25
- Large case (200+ documents): $0.25-$1.00

## Error Handling

- Graceful handling of corrupted files
- OCR fallback for unreadable PDFs
- Detailed logging for troubleshooting
- Automatic cleanup of temporary files

## Security

- Temporary files are automatically cleaned up
- No data is stored permanently on server
- API keys are handled securely via environment variables

## Troubleshooting

1. **"No OpenAI API key found"**: Set the OPENAI_API_KEY environment variable
2. **OCR not working**: Install Tesseract OCR and ensure it's in your PATH
3. **Large files failing**: Enable auto-split option
4. **Memory errors**: Process smaller batches of files

## Legal Notice

This tool is designed to assist legal professionals in analyzing medical records. Always verify AI-generated summaries against original documents. The tool does not provide legal advice.