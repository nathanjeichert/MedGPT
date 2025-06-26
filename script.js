// Medical Records Summary Tool - JavaScript Functionality

class MedicalRecordsProcessor {
    constructor() {
        this.selectedFiles = [];
        this.processing = false;
        this.currentResultId = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.setupFormValidation();
    }

    bindEvents() {
        // Form submission
        const form = document.getElementById('medicalRecordsForm');
        form.addEventListener('submit', (e) => this.handleSubmit(e));

        // File input change
        const folderInput = document.getElementById('folderInput');
        folderInput.addEventListener('change', (e) => this.handleFolderSelection(e));

        // Clear button
        const clearBtn = document.getElementById('clearBtn');
        clearBtn.addEventListener('click', () => this.clearForm());

        // Download and view buttons
        const downloadJsonBtn = document.getElementById('downloadJsonBtn');
        const downloadOverviewBtn = document.getElementById('downloadOverviewBtn');
        const downloadChronologyBtn = document.getElementById('downloadChronologyBtn');
        const viewBtn = document.getElementById('viewBtn');
        
        downloadJsonBtn.addEventListener('click', () => this.downloadJsonResults());
        downloadOverviewBtn.addEventListener('click', () => this.downloadOverviewDoc());
        downloadChronologyBtn.addEventListener('click', () => this.downloadChronologyDoc());
        viewBtn.addEventListener('click', () => this.viewResults());
    }

    setupFormValidation() {
        const folderInput = document.getElementById('folderInput');
        const clientName = document.getElementById('clientName');
        const casePrompt = document.getElementById('casePrompt');
        const processBtn = document.getElementById('processBtn');

        const validateForm = () => {
            const hasFiles = this.selectedFiles.length > 0;
            const hasClientName = clientName.value.trim().length > 0;
            const hasPrompt = casePrompt.value.trim().length > 0;
            
            processBtn.disabled = !hasFiles || !hasClientName || !hasPrompt || this.processing;
        };

        folderInput.addEventListener('change', validateForm);
        clientName.addEventListener('input', validateForm);
        casePrompt.addEventListener('input', validateForm);
        
        // Initial validation
        validateForm();
    }

    handleFolderSelection(event) {
        const files = Array.from(event.target.files);
        this.selectedFiles = files;

        const selectedFolderDiv = document.getElementById('selectedFolder');
        const fileInputDisplay = document.querySelector('.file-input-display .file-text');

        if (files.length > 0) {
            // Get folder name from first file path
            const firstFile = files[0];
            const pathParts = firstFile.webkitRelativePath.split('/');
            const folderName = pathParts[0];
            
            // Count file types
            const fileTypes = this.categorizeFiles(files);
            
            selectedFolderDiv.innerHTML = `
                <strong>Selected Folder:</strong> ${folderName}<br>
                <strong>Total Files:</strong> ${files.length}<br>
                <strong>File Types:</strong> ${this.formatFileTypes(fileTypes)}
            `;
            selectedFolderDiv.classList.add('show');
            
            fileInputDisplay.textContent = `${files.length} files selected from "${folderName}"`;
        } else {
            selectedFolderDiv.classList.remove('show');
            fileInputDisplay.textContent = 'Click to select folder';
        }
    }

    categorizeFiles(files) {
        const types = {};
        files.forEach(file => {
            const ext = file.name.split('.').pop().toLowerCase();
            types[ext] = (types[ext] || 0) + 1;
        });
        return types;
    }

    formatFileTypes(fileTypes) {
        return Object.entries(fileTypes)
            .map(([ext, count]) => `${ext.toUpperCase()}: ${count}`)
            .join(', ');
    }

    async handleSubmit(event) {
        event.preventDefault();
        
        if (this.processing) {
            return;
        }

        this.processing = true;
        this.showProcessingState();

        try {
            const formData = this.collectFormData();
            await this.processFiles(formData);
        } catch (error) {
            console.error('Processing error:', error);
            this.showError('An error occurred during processing: ' + error.message);
        } finally {
            this.processing = false;
            this.hideProcessingState();
        }
    }

    collectFormData() {
        const clientName = document.getElementById('clientName').value.trim();
        const casePrompt = document.getElementById('casePrompt').value.trim();
        const autoSplit = document.getElementById('autoSplit').checked;
        const generateLawyerDocs = document.getElementById('generateLawyerDocs').checked;

        return {
            files: this.selectedFiles,
            clientName: clientName,
            casePrompt: casePrompt,
            autoSplit: autoSplit,
            generateLawyerDocs: generateLawyerDocs
        };
    }

    async processFiles(formData) {
        console.log('Preparing files for upload...');
        
        try {
            // Step 1: Create upload session and get signed URLs
            const filesInfo = formData.files.map(file => ({
                name: file.name,
                type: file.type,
                size: file.size
            }));
            
            console.log('Creating upload session...');
            
            const sessionResponse = await fetch('/api/upload-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ files: filesInfo })
            });
            
            if (!sessionResponse.ok) {
                throw new Error('Failed to create upload session');
            }
            
            const sessionData = await sessionResponse.json();
            const { session_id, upload_urls } = sessionData;
            
            // Step 2: Upload files directly to Cloud Storage
            console.log('Uploading files to local storage...');
            
            const uploadPromises = upload_urls.map(async (urlInfo, index) => {
                const file = formData.files[index];
                const response = await fetch(urlInfo.upload_url, {
                    method: 'PUT',
                    body: file,
                    headers: {
                        'Content-Type': file.type
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to upload ${file.name}`);
                }
                
                return urlInfo.filename;
            });
            
            await Promise.all(uploadPromises);
            console.log('Files uploaded, starting AI processing...');
            
            // Step 3: Start processing with Cloud Storage files
            const processResponse = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: session_id,
                    clientName: formData.clientName,
                    casePrompt: formData.casePrompt,
                    autoSplit: formData.autoSplit,
                    generateLawyerDocs: formData.generateLawyerDocs
                })
            });
            
            if (!processResponse.ok) {
                let errorMessage = 'Processing failed';
                try {
                    const contentType = processResponse.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        const errorData = await processResponse.json();
                        errorMessage = errorData.error || errorMessage;
                    } else {
                        const errorText = await processResponse.text();
                        console.error('Non-JSON error response:', errorText);
                        errorMessage = `Server error (${processResponse.status}): ${processResponse.statusText}`;
                    }
                } catch (parseError) {
                    console.error('Failed to parse error response:', parseError);
                    errorMessage = `Server error (${processResponse.status}): ${processResponse.statusText}`;
                }
                throw new Error(errorMessage);
            }
            
            const result = await processResponse.json();
            
            if (result.success && result.task_id) {
                // Handle real processing results
                if (result.result) {
                    this.currentResultId = result.result.result_id;
                    this.showResults(result.result);
                } else {
                    alert('Processing started successfully!');
                }
            } else {
                throw new Error(result.error || 'Failed to start processing');
            }
            
        } catch (error) {
            console.error('Processing error:', error);
            this.showError(error.message);
        }
    }

    showCompletionMessage() {
        // Simple completion handler for local development
        const resultsSection = document.getElementById('resultsSection');
        const downloadSection = document.getElementById('downloadSection');
        
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }
        
        if (downloadSection) {
            downloadSection.style.display = 'block';
        }
        
        // Show a simple alert instead of undefined function
        alert('Processing complete! Files have been processed and are ready for download.');
    }

    generateMockResults(formData) {
        const totalFiles = formData.files.length;
        const estimatedRecords = Math.floor(totalFiles * 2.5); // Estimate records per file
        
        return {
            totalFiles: totalFiles,
            totalRecords: estimatedRecords,
            processingTime: '5 minutes 32 seconds',
            fileSize: '2.4 MB',
            wordCount: 45000,
            tokensUsed: 125000,
            estimatedCost: '$0.0125',
            splitFiles: formData.autoSplit && estimatedRecords > 100 ? 2 : 1
        };
    }

    // Progress functions removed for local development

    showProcessingState() {
        const processBtn = document.getElementById('processBtn');
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');
        
        processBtn.disabled = true;
        btnText.textContent = 'Processing...';
        btnSpinner.style.display = 'inline-block';
    }

    hideProcessingState() {
        const processBtn = document.getElementById('processBtn');
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');

        processBtn.disabled = false;
        btnText.textContent = 'Process Medical Records';
        btnSpinner.style.display = 'none';
    }

    showResults(stats) {
        const resultsSection = document.getElementById('resultsSection');
        const summaryStats = document.getElementById('summaryStats');

        summaryStats.innerHTML = `
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Total Files Processed:</span>
                    <span class="stat-value">${stats.total_files}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Medical Records Extracted:</span>
                    <span class="stat-value">${stats.total_records}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Records Relevant to Case:</span>
                    <span class="stat-value">${stats.relevant_records}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">AI Tokens Used:</span>
                    <span class="stat-value">${stats.tokens_used.toLocaleString()}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Estimated Cost:</span>
                    <span class="stat-value">${stats.estimated_cost}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Lawyer Documents:</span>
                    <span class="stat-value">${stats.lawyer_docs_generated} generated</span>
                </div>
                ${stats.split_files > 1 ? `
                <div class="stat-item">
                    <span class="stat-label">Split into Files:</span>
                    <span class="stat-value">${stats.split_files} parts</span>
                </div>
                ` : ''}
            </div>
        `;

        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    downloadJsonResults() {
        if (!this.currentResultId) {
            alert('No results available for download');
            return;
        }
        
        window.open(`/api/download/${this.currentResultId}/json`, '_blank');
    }

    downloadOverviewDoc() {
        if (!this.currentResultId) {
            alert('No results available for download');
            return;
        }
        
        window.open(`/api/download/${this.currentResultId}/overview`, '_blank');
    }

    downloadChronologyDoc() {
        if (!this.currentResultId) {
            alert('No results available for download');
            return;
        }
        
        window.open(`/api/download/${this.currentResultId}/chronology`, '_blank');
    }

    viewResults() {
        // In a real implementation, this would open a results viewer
        const mockData = {
            medical_records_summary: "Sample Medical Records Analysis",
            processed_date: new Date().toISOString(),
            total_files: 15,
            total_records: 37
        };

        const newWindow = window.open('', '_blank');
        newWindow.document.write(`
            <html>
                <head>
                    <title>Medical Records Summary Results</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }
                        .header { background: #2563eb; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
                        .stat { background: #f8fafc; padding: 10px; margin: 10px 0; border-left: 4px solid #2563eb; }
                        pre { background: #f1f5f9; padding: 15px; border-radius: 5px; overflow: auto; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Medical Records Summary Results</h1>
                        <p>Processing completed on ${new Date().toLocaleString()}</p>
                    </div>
                    <div class="stat">
                        <strong>Total Files Processed:</strong> ${mockData.total_files}
                    </div>
                    <div class="stat">
                        <strong>Total Records Extracted:</strong> ${mockData.total_records}
                    </div>
                    <h2>Sample Data Structure:</h2>
                    <pre>${JSON.stringify(mockData, null, 2)}</pre>
                    <p><em>Note: This is a demo showing the expected output format. In the actual implementation, full medical records data would be displayed here.</em></p>
                </body>
            </html>
        `);
    }

    showError(message) {
        const progressSection = document.getElementById('progressSection');
        const progressText = document.getElementById('progressText');
        const progressDetails = document.getElementById('progressDetails');

        progressText.innerHTML = `<span style="color: var(--error-color)">❌ Error: ${message}</span>`;
        progressDetails.textContent = 'Please check your inputs and try again.';
    }

    clearForm() {
        // Reset form fields
        document.getElementById('folderInput').value = '';
        document.getElementById('clientName').value = '';
        document.getElementById('casePrompt').value = '';
        document.getElementById('autoSplit').checked = true;
        document.getElementById('generateLawyerDocs').checked = true;

        // Reset state
        this.selectedFiles = [];
        this.processing = false;
        
        // Clean up previous results if they exist
        if (this.currentResultId) {
            fetch(`/api/cleanup/${this.currentResultId}`, { method: 'POST' });
            this.currentResultId = null;
        }

        // Hide sections
        document.getElementById('selectedFolder').classList.remove('show');
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';

        // Reset display text
        document.querySelector('.file-input-display .file-text').textContent = 'Click to select folder';

        // Reset button state
        this.hideProcessingState();
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MedicalRecordsProcessor();
});

// Add CSS for stats grid
const additionalCSS = `
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.stat-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: var(--background-color);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
}

.stat-label {
    font-weight: 500;
    color: var(--text-secondary);
}

.stat-value {
    font-weight: 600;
    color: var(--text-primary);
}
`;

// Inject additional CSS
const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);