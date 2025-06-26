#!/usr/bin/env python3
"""
Flask server for Medical Records Summary Tool
"""

import os
import json
import tempfile
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template_string, Response
from werkzeug.utils import secure_filename
import zipfile

# Import our medical records processing
from ingest import MedicalRecordsEngine, LawyerDocumentGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024  # 50GB max upload

# Check for OpenAI API key at startup
if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: Please set OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-api-key-here'")
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

# Store results and progress temporarily
results_store = {}
progress_store = {}

@app.route('/')
def index():
    """Serve the main HTML page."""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({'error': 'index.html not found'}), 500

@app.route('/styles.css')
def styles():
    """Serve CSS file."""
    try:
        with open('styles.css', 'r') as f:
            return f.read(), 200, {'Content-Type': 'text/css'}
    except FileNotFoundError:
        return jsonify({'error': 'styles.css not found'}), 404

@app.route('/script.js')
def script():
    """Serve JavaScript file."""
    try:
        with open('script.js', 'r') as f:
            return f.read(), 200, {'Content-Type': 'application/javascript'}
    except FileNotFoundError:
        return jsonify({'error': 'script.js not found'}), 404

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with JSON response."""
    return jsonify({'error': 'Not found', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with JSON response."""
    return jsonify({'error': 'Internal server error', 'status': 500}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Cloud Run."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    """Get processing progress via Server-Sent Events."""
    def generate():
        # Check if task exists, if not send error and exit
        if task_id not in progress_store:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Task not found'})}\n\n"
            return
            
        while task_id in progress_store:
            progress = progress_store[task_id]
            yield f"data: {json.dumps(progress)}\n\n"
            
            if progress.get('status') == 'completed' or progress.get('status') == 'error':
                break
                
            time.sleep(0.5)  # Update every 500ms
    
    return Response(generate(), mimetype='text/event-stream')

def process_medical_records_async(task_id, form_data, files):
    """Process medical records asynchronously with progress tracking."""
    try:
        progress_store[task_id] = {'progress': 0, 'status': 'starting', 'message': 'Initializing...'}
        
        # Get form data
        client_name = form_data.get('clientName', 'Client')
        case_prompt = form_data.get('casePrompt', '')
        auto_split = form_data.get('autoSplit') == 'true'
        generate_lawyer_docs = form_data.get('generateLawyerDocs') == 'true'
        
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            progress_store[task_id] = {'progress': 0, 'status': 'error', 'message': 'OpenAI API key not found'}
            return
        
        progress_store[task_id] = {'progress': 10, 'status': 'processing', 'message': 'Creating temporary directory...'}
        
        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            progress_store[task_id] = {'progress': 15, 'status': 'processing', 'message': 'Saving uploaded files...'}
            
            # Save uploaded files
            if not files:
                progress_store[task_id] = {'progress': 0, 'status': 'error', 'message': 'No files uploaded'}
                return
            
            total_files = len(files)
            for i, file in enumerate(files):
                if file.filename:
                    # Create subdirectories based on file path
                    file_path = secure_filename(file.filename)
                    full_path = temp_dir / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    file.save(str(full_path))
                    
                    # Update progress for file saving
                    progress = 15 + (i + 1) / total_files * 10  # 15-25%
                    progress_store[task_id] = {
                        'progress': int(progress), 
                        'status': 'processing', 
                        'message': f'Saved file {i+1}/{total_files}: {file.filename}'
                    }
            
            progress_store[task_id] = {'progress': 30, 'status': 'processing', 'message': 'Starting medical records processing...'}
            
            # Process medical records with progress tracking
            engine = MedicalRecordsEngine(api_key)
            
            # Custom processing with progress tracking
            summary = process_case_with_progress(engine, temp_dir, case_prompt, task_id)
            
            progress_store[task_id] = {'progress': 85, 'status': 'processing', 'message': 'Generating lawyer documents...'}
            
            # Generate lawyer documents if requested
            lawyer_docs = []
            if generate_lawyer_docs:
                doc_generator = LawyerDocumentGenerator(api_key)
                lawyer_docs = doc_generator.generate_lawyer_documents(
                    summary, client_name, case_prompt, temp_dir
                )
                progress_store[task_id] = {'progress': 92, 'status': 'processing', 'message': f'Generated {len(lawyer_docs)} lawyer documents'}
            
            # Auto-split if requested
            split_files = []
            summary_path = temp_dir / '_medical_records_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            progress_store[task_id] = {'progress': 95, 'status': 'processing', 'message': 'Checking file splitting...'}
            
            if auto_split:
                split_files = engine.split_json_file(summary_path)
            else:
                split_files = [summary_path]
            
            # Store results with unique ID
            import uuid
            result_id = str(uuid.uuid4())
            
            results_store[result_id] = {
                'summary': summary,
                'lawyer_docs': lawyer_docs,
                'split_files': split_files,
                'temp_dir': temp_dir,
                'client_name': client_name
            }
            
            # Calculate statistics
            total_files = len(summary.get('document_registry', {}))
            total_records = sum(len(doc.get('records', [])) for doc in summary.get('document_registry', {}).values())
            relevant_records = sum(1 for doc in summary.get('document_registry', {}).values() 
                                 for record in doc.get('records', []) 
                                 if record.get('mentions_target_injury', False))
            
            progress_store[task_id] = {
                'progress': 100, 
                'status': 'completed', 
                'message': 'Processing complete!',
                'result_id': result_id,
                'stats': {
                    'total_files': total_files,
                    'total_records': total_records,
                    'relevant_records': relevant_records,
                    'lawyer_docs_generated': len(lawyer_docs),
                    'split_files': len(split_files),
                    'tokens_used': summary.get('total_case_tokens', 0),
                    'estimated_cost': summary.get('estimated_cost', 'N/A')
                }
            }
            
        except Exception as e:
            # Clean up temp directory on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            progress_store[task_id] = {'progress': 0, 'status': 'error', 'message': str(e)}
            
    except Exception as e:
        progress_store[task_id] = {'progress': 0, 'status': 'error', 'message': str(e)}

def process_case_with_progress(engine, case_path, case_prompt, task_id):
    """Modified version of process_case that tracks progress."""
    # This is a simplified version - you could enhance this further
    # by modifying the actual engine to provide progress callbacks
    
    progress_store[task_id] = {'progress': 35, 'status': 'processing', 'message': 'Analyzing folder structure...'}
    
    # Use the existing process_case method but update progress at key points
    summary = engine.process_case(case_path, case_prompt)
    
    progress_store[task_id] = {'progress': 80, 'status': 'processing', 'message': 'AI processing complete, finalizing results...'}
    
    return summary

@app.route('/api/process', methods=['POST'])
def process_medical_records():
    """Start processing uploaded medical records asynchronously."""
    try:
        # Get form data
        form_data = {
            'clientName': request.form.get('clientName', 'Client'),
            'casePrompt': request.form.get('casePrompt', ''),
            'autoSplit': request.form.get('autoSplit'),
            'generateLawyerDocs': request.form.get('generateLawyerDocs')
        }
        
        # Get uploaded files
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        # Generate task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_medical_records_async,
            args=(task_id, form_data, files)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<result_id>/<doc_type>')
def download_result(result_id, doc_type):
    """Download processed results."""
    if result_id not in results_store:
        return jsonify({'error': 'Results not found'}), 404
    
    result = results_store[result_id]
    temp_dir = result['temp_dir']
    client_name = result['client_name']
    
    try:
        if doc_type == 'json':
            # Return JSON summary
            json_files = result['split_files']
            if len(json_files) == 1:
                return send_file(json_files[0], as_attachment=True, download_name='medical_records_summary.json')
            else:
                # Create zip file with all parts
                zip_path = temp_dir / 'medical_records_summary.zip'
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for json_file in json_files:
                        zf.write(json_file, json_file.name)
                return send_file(zip_path, as_attachment=True, download_name='medical_records_summary.zip')
        
        elif doc_type == 'overview':
            # Return medical overview Word doc
            lawyer_docs = result['lawyer_docs']
            overview_doc = next((doc for doc in lawyer_docs if 'Overview' in doc.name), None)
            if overview_doc and overview_doc.exists():
                return send_file(overview_doc, as_attachment=True, download_name=f'AI Medical Records Overview - {client_name}.docx')
            else:
                return jsonify({'error': 'Overview document not found'}), 404
        
        elif doc_type == 'chronology':
            # Return medical chronology Word doc
            lawyer_docs = result['lawyer_docs']
            chronology_doc = next((doc for doc in lawyer_docs if 'Chronology' in doc.name), None)
            if chronology_doc and chronology_doc.exists():
                return send_file(chronology_doc, as_attachment=True, download_name=f'AI Med. Chronology - {client_name}.docx')
            else:
                return jsonify({'error': 'Chronology document not found'}), 404
        
        else:
            return jsonify({'error': 'Invalid document type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<result_id>', methods=['POST'])
def cleanup_results(result_id):
    """Clean up temporary files."""
    if result_id in results_store:
        result = results_store[result_id]
        temp_dir = result['temp_dir']
        shutil.rmtree(temp_dir, ignore_errors=True)
        del results_store[result_id]
        return jsonify({'success': True})
    
    return jsonify({'error': 'Results not found'}), 404

@app.route('/api/cleanup-progress/<task_id>', methods=['POST'])
def cleanup_progress(task_id):
    """Clean up progress tracking data."""
    if task_id in progress_store:
        del progress_store[task_id]
    
    return jsonify({'success': True})

if __name__ == '__main__':
    # Check for OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Get port from environment (Cloud Run sets this)
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting Medical Records Summary Tool Server...")
    print(f"Server will run on port {port}")
    print("Make sure you have set OPENAI_API_KEY environment variable")
    
    # Use debug=False for production Cloud Run deployment
    app.run(debug=False, host='0.0.0.0', port=port)