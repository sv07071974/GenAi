import os
import requests
from bs4 import BeautifulSoup
import openai
import pandas as pd
import concurrent.futures
import time
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory

app = Flask(__name__)

# Create a temp directory for CSV files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Dictionary to store CSV file paths
csv_files = {}

# Get API key from environment variable (if set)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Function to format URL properly
def format_url(url):
    # Strip whitespace
    url = url.strip()

    # Add https:// if no protocol specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Add trailing slash if not present
    if not url.endswith('/'):
        url = url + '/'

    return url

# Function to extract text from website
def extract_text_from_url(url):
    try:
        # Format the URL before use
        formatted_url = format_url(url)

        # Send request to get website content
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(formatted_url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text(separator='\n')

        # Clean text (remove extra whitespace and empty lines)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except requests.exceptions.RequestException as e:
        return f"Error fetching website: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to translate text to English using OpenAI
def translate_text(text, api_key):
    if not api_key:
        return "Please provide an OpenAI API key", text

    try:
        client = openai.OpenAI(api_key=api_key)

        # If text is too long, process it in chunks
        max_chunk_size = 4000  # Adjust based on token limits
        if len(text) > max_chunk_size:
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []

            for chunk in chunks:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful translation assistant. Translate the following text to English while preserving formatting."},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.3
                )
                translated_chunks.append(response.choices[0].message.content)

            translated_text = "".join(translated_chunks)
        else:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful translation assistant. Translate the following text to English while preserving formatting."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            translated_text = response.choices[0].message.content

        return "Translation successful", translated_text
    except Exception as e:
        return f"Translation error: {str(e)}", text

# Function to generate company description using OpenAI
def generate_company_description(text, url, api_key):
    if not api_key:
        return "API key required for company description"

    try:
        client = openai.OpenAI(api_key=api_key)

        # Use first 3000 characters of text to get context about company
        context = text[:3000]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst assistant. Based on the website content provided, give a concise 15-word description of what the company or website is about."},
                {"role": "user", "content": f"Website: {url}\n\nContent sample: {context}\n\nProvide a concise 15-word description of what this company or website is about:"}
            ],
            max_tokens=50,
            temperature=0.7
        )

        description = response.choices[0].message.content.strip()

        # Ensure description is around 15 words
        words = description.split()
        if len(words) > 20:  # Allow some flexibility
            description = " ".join(words[:15]) + "..."

        return description
    except Exception as e:
        return f"Could not generate description: {str(e)}"

# Function to check if any of the comma-separated terms exist in website content
def check_text_in_website(url, search_terms_text, api_key):
    # Format the URL for display
    formatted_url = format_url(url)

    # Extract text from website
    extracted_text = extract_text_from_url(url)

    if isinstance(extracted_text, str) and extracted_text.startswith("Error"):
        return {"Website": formatted_url, "Status": "Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": extracted_text}

    # Translate text to English
    translation_status, translated_text = translate_text(extracted_text, api_key)

    if translation_status.startswith("Translation error"):
        return {"Website": formatted_url, "Status": "Translation Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": translation_status}

    # Generate company description
    company_description = generate_company_description(translated_text, formatted_url, api_key)

    # Parse comma-separated search terms
    search_terms = [term.strip().lower() for term in search_terms_text.split(',') if term.strip()]

    # Initialize result dictionary for each term
    term_results = {}
    any_term_found = False

    for term in search_terms:
        # Check if term is in translated content (case insensitive)
        contains_term = "Yes" if term.lower() in translated_text.lower() else "No"
        term_results[term] = contains_term

        if contains_term == "Yes":
            any_term_found = True

    # Determine decision
    decision = "Rejected" if any_term_found else "Accepted"

    # Add basic information to result
    result = {
        "Website": formatted_url,
        "Status": "Success",
        "Company Description": company_description,
        "Decision": decision,
        "Error Message": ""
    }

    # Add result for each search term
    for term, contains in term_results.items():
        result[f"Contains '{term}'"] = contains

    return result

# API route for analyzing websites
@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze_websites():
    # Add CORS headers for preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        return ('', 204, headers)
        
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    urls_text = data.get('urls', '')
    search_terms_text = data.get('searchTerms', '')
    api_key = data.get('apiKey', DEFAULT_API_KEY)
    
    if not urls_text or not search_terms_text:
        return jsonify({"error": "Please provide both website URLs and search terms"}), 400
    
    if not api_key:
        return jsonify({"error": "Please provide an OpenAI API key"}), 400
    
    # Parse list of URLs
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    if not urls:
        return jsonify({"error": "No valid URLs provided"}), 400
    
    results = []
    
    # Process each URL in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(check_text_in_website, url, search_terms_text, api_key): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            results.append(result)
    
    # Create DataFrame for CSV export
    df = pd.DataFrame(results)
    
    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Save CSV file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = f"website_analysis_{timestamp}.csv"
    csv_path = os.path.join(TEMP_DIR, csv_filename)
    df.to_csv(csv_path, index=False)
    
    # Store the CSV path
    csv_files[analysis_id] = csv_path
    
    response = jsonify({
        "status": "success",
        "message": "Analysis complete",
        "results": results,
        "analysisId": analysis_id
    })
    
    # Add CORS headers to response
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# API route for downloading CSV
@app.route('/api/download/<analysis_id>', methods=['GET'])
def download_csv(analysis_id):
    if analysis_id not in csv_files:
        return jsonify({"error": "Analysis not found"}), 404
    
    csv_path = csv_files[analysis_id]
    
    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not found"}), 404
    
    # Get the filename from the path
    filename = os.path.basename(csv_path)
    
    # Send the file to the client
    return send_file(csv_path, as_attachment=True, download_name=filename)

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

# Root route to serve the HTML interface
@app.route('/')
def index():
    return render_template('index.html')

# Favicon handler
@app.route('/favicon.ico')
def favicon():
    return "", 204

# Define HTML template as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Website Text Analyzer</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3f37c9;
            --success-color: #4cc9f0;
            --danger-color: #f72585;
            --warning-color: #f9c74f;
            --info-color: #90e0ef;
            --light-color: #f8f9fa;
            --dark-color: #212529;
        }
        
        body {
            background-color: #f5f7fb;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .navbar {
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .navbar-brand {
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        
        .container {
            max-width: 1280px;
        }
        
        .card {
            border-radius: 1rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
            border: none;
            margin-bottom: 30px;
            background-color: white;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card-header {
            background-color: white;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            font-weight: 700;
            padding: 1.25rem 1.5rem;
            border-radius: 1rem 1rem 0 0 !important;
        }
        
        .card-body {
            padding: 1.5rem;
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
            border-radius: 0.5rem;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            background-color: var(--secondary-color);
            border-color: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .btn-success {
            background-color: var(--success-color);
            border-color: var(--success-color);
            color: white;
            border-radius: 0.5rem;
            padding: 0.4rem 1rem;
            font-weight: 600;
        }
        
        .btn-success:hover {
            background-color: #3ab7db;
            border-color: #3ab7db;
        }
        
        .form-control {
            border-radius: 0.5rem;
            padding: 0.6rem 1rem;
            border: 1px solid #e0e0e0;
            background-color: #f8fafc;
        }
        
        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 0.2rem rgba(67, 97, 238, 0.15);
        }
        
        .form-label {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #333;
        }
        
        .nav-tabs {
            border-bottom: 2px solid #f0f0f0;
        }
        
        .nav-tabs .nav-link {
            border: none;
            color: #6c757d;
            padding: 0.75rem 1.25rem;
            font-weight: 600;
            border-radius: 0.5rem 0.5rem 0 0;
        }
        
        .nav-tabs .nav-link:hover {
            color: var(--primary-color);
        }
        
        .nav-tabs .nav-link.active {
            color: var(--primary-color);
            background-color: transparent;
            border-bottom: 3px solid var(--primary-color);
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .loading-overlay.visible {
            visibility: visible;
            opacity: 1;
        }
        
        .spinner-container {
            text-align: center;
            background-color: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        
        .spinner-border {
            color: var(--primary-color);
        }
        
        .table-container {
            max-height: 600px;
            overflow-y: auto;
            border-radius: 0.5rem;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.05);
        }
        
        .results-table {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
        }
        
        .results-table th {
            position: sticky;
            top: 0;
            background-color: #f8f9fa;
            z-index: 10;
            padding: 1rem;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #e9ecef;
        }
        
        .results-table td {
            padding: 1rem;
            vertical-align: middle;
            border-bottom: 1px solid #e9ecef;
        }
        
        .results-table tr:last-child td {
            border-bottom: none;
        }
        
        .badge {
            padding: 0.4rem 0.8rem;
            border-radius: 0.5rem;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .badge-success {
            background-color: rgba(76, 201, 240, 0.15);
            color: #4cc9f0;
            border: 1px solid rgba(76, 201, 240, 0.3);
        }
        
        .badge-danger {
            background-color: rgba(247, 37, 133, 0.15);
            color: #f72585;
            border: 1px solid rgba(247, 37, 133, 0.3);
        }
        
        .accepted {
            background-color: rgba(76, 201, 240, 0.05);
        }
        
        .rejected {
            background-color: rgba(247, 37, 133, 0.05);
        }
        
        .instructions-list {
            padding-left: 1.5rem;
        }
        
        .instructions-list li {
            margin-bottom: 0.75rem;
            position: relative;
        }
        
        .instructions-list ul {
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        }
        
        .form-text {
            color: #6c757d;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        .input-group-text {
            background-color: #f8fafc;
            border: 1px solid #e0e0e0;
            border-left: none;
            border-radius: 0 0.5rem 0.5rem 0;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f5f5f5;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #999;
        }
    </style>
</head>
<body>
    <!-- Loading Overlay -->
    <div id="loadingOverlay" class="loading-overlay">
        <div class="spinner-container">
            <div class="spinner-border" role="status" style="width: 3rem; height: 3rem;"></div>
            <p class="mt-3 mb-0 fw-bold">Processing websites...</p>
            <p class="text-muted">This may take a few minutes.</p>
        </div>
    </div>

    <!-- Main Content -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-globe me-2"></i>Multi-Website Text Analyzer
            </a>
        </div>
    </nav>

    <div class="container py-5">
        <!-- Input Form -->
        <div class="card mb-4">
            <div class="card-header d-flex align-items-center">
                <i class="fas fa-search me-2 text-primary"></i>
                <span>Website Analysis</span>
            </div>
            <div class="card-body">
                <ul class="nav nav-tabs mb-4" id="inputTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="urls-tab" data-bs-toggle="tab" data-bs-target="#urls-tab-pane" type="button" role="tab">
                            <i class="fas fa-link me-2"></i>Enter URLs
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="file-tab" data-bs-toggle="tab" data-bs-target="#file-tab-pane" type="button" role="tab">
                            <i class="fas fa-file-upload me-2"></i>Upload URL File
                        </button>
                    </li>
                </ul>
                
                <div class="tab-content" id="inputTabsContent">
                    <div class="tab-pane fade show active" id="urls-tab-pane" role="tabpanel" tabindex="0">
                        <div class="mb-4">
                            <label for="urlsInput" class="form-label">Website URLs (one per line)</label>
                            <textarea class="form-control" id="urlsInput" rows="6" placeholder="example.com&#10;another-site.org"></textarea>
                            <div class="form-text">
                                <i class="fas fa-info-circle me-1"></i>
                                URLs will automatically have 'https://' added if missing and '/' added at the end
                            </div>
                        </div>
                    </div>
                    <div class="tab-pane fade" id="file-tab-pane" role="tabpanel" tabindex="0">
                        <div class="mb-4">
                            <label for="fileInput" class="form-label">Upload a text file with URLs</label>
                            <input class="form-control" type="file" id="fileInput" accept=".txt">
                        </div>
                        <div class="mb-4">
                            <label for="fileUrlsPreview" class="form-label">URLs from file</label>
                            <textarea class="form-control" id="fileUrlsPreview" rows="6" readonly></textarea>
                        </div>
                    </div>
                </div>
                
                <div class="row g-4">
                    <div class="col-md-6">
                        <label for="searchTermsInput" class="form-label">Terms to Search For (comma-separated)</label>
                        <input type="text" class="form-control" id="searchTermsInput" placeholder="term1, term2, term3">
                    </div>
                    <div class="col-md-6">
                        <label for="apiKeyInput" class="form-label">OpenAI API Key</label>
                        <div class="input-group">
                            <input type="password" class="form-control" id="apiKeyInput" placeholder="sk-...">
                            <button class="input-group-text" type="button" id="toggleApiKey">
                                <i class="fa fa-eye"></i>
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="mt-4 text-center">
    <button id="analyzeButton" class="btn btn-primary" onclick="(function() {
        // Get input values
        const urlsInput = document.getElementById('urlsInput');
        const fileUrlsPreview = document.getElementById('fileUrlsPreview');
        const searchTermsInput = document.getElementById('searchTermsInput');
        const apiKeyInput = document.getElementById('apiKeyInput');
        
        const urls = (urlsInput ? urlsInput.value.trim() : '') || 
                     (fileUrlsPreview ? fileUrlsPreview.value.trim() : '');
        const searchTerms = searchTermsInput ? searchTermsInput.value.trim() : '';
        const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
        
        // Validate inputs
        if (!urls) {
            alert('Please enter at least one URL');
            return;
        }
        
        if (!searchTerms) {
            alert('Please enter at least one search term');
            return;
        }
        
        if (!apiKey) {
            alert('Please enter your OpenAI API key');
            return;
        }
        
        // Show loading overlay
        document.getElementById('loadingOverlay').style.visibility = 'visible';
        
        // Make API request
        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                urls: urls,
                searchTerms: searchTerms,
                apiKey: apiKey
            }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to analyze websites');
                });
            }
            return response.json();
        })
        .then(data => {
            window.currentResults = data.results;
            window.currentAnalysisId = data.analysisId;
            
            // Define the display function inline
            function displayResults(results) {
                if (!results || results.length === 0) return;
                
                // Get all possible columns
                const allColumns = new Set();
                results.forEach(result => {
                    Object.keys(result).forEach(key => allColumns.add(key));
                });
                
                // Filter out specific columns to display first
                const priorityColumns = ['Website', 'Status', 'Company Description', 'Decision'];
                
                // Get search term columns
                const searchTermColumns = Array.from(allColumns)
                    .filter(col => col.startsWith('Contains \\''))
                    .sort();
                
                // Get error message column
                const errorMessageColumn = 'Error Message';
                
                // Combine columns in the desired order
                const columns = [
                    ...priorityColumns,
                    ...searchTermColumns,
                    errorMessageColumn
                ];
                
                // Create table header
                const resultsTableHeader = document.getElementById('resultsTableHeader');
                if (resultsTableHeader) {
                    resultsTableHeader.innerHTML = '';
                    columns.forEach(column => {
                        const th = document.createElement('th');
                        th.textContent = column;
                        resultsTableHeader.appendChild(th);
                    });
                }
                
                // Create table rows
                const resultsTableBody = document.getElementById('resultsTableBody');
                if (resultsTableBody) {
                    resultsTableBody.innerHTML = '';
                    results.forEach(result => {
                        const tr = document.createElement('tr');
                        
                        // Add decision-based row class
                        if (result.Decision === 'Accepted') {
                            tr.classList.add('accepted');
                        } else if (result.Decision === 'Rejected') {
                            tr.classList.add('rejected');
                        }
                        
                        columns.forEach(column => {
                            const td = document.createElement('td');
                            
                            // Format specific columns
                            if (column === 'Decision') {
                                const badge = document.createElement('span');
                                badge.className = 'badge ' + (result[column] === 'Accepted' ? 'badge-success' : 'badge-danger');
                                badge.textContent = result[column] || 'N/A';
                                td.appendChild(badge);
                            } else if (column.startsWith('Contains \\'')) {
                                const badge = document.createElement('span');
                                badge.className = 'badge ' + (result[column] === 'Yes' ? 'badge-danger' : 'badge-success');
                                badge.textContent = result[column] || 'N/A';
                                td.appendChild(badge);
                            } else if (column === 'Website') {
                                // Make website a clickable link
                                const link = document.createElement('a');
                                link.href = result[column];
                                link.target = '_blank'; // Open in new tab
                                link.textContent = result[column];
                                link.className = 'text-decoration-none';
                                td.appendChild(link);
                            } else {
                                td.textContent = result[column] || 'N/A';
                            }
                            
                            tr.appendChild(td);
                        });
                        
                        resultsTableBody.appendChild(tr);
                    });
                }
            }
            
            // Call the inline display function
            displayResults(data.results);
            
            // Show results card, hide instructions
            document.getElementById('resultsCard').style.display = 'block';
            document.getElementById('instructionsCard').style.display = 'none';
            document.getElementById('resultsCard').scrollIntoView({behavior: 'smooth'});
        })
        .catch(error => {
            console.error('Error analyzing websites:', error);
            alert('Error: ' + error.message);
        })
        .finally(() => {
            // Hide loading overlay
            document.getElementById('loadingOverlay').style.visibility = 'hidden';
        });
    })()">
        <i class="fas fa-play me-2"></i>Check Websites
    </button>
</div>
            </div>
        </div>

        <!-- Instructions -->
        <div class="card" id="instructionsCard">
            <div class="card-header d-flex align-items-center">
                <i class="fas fa-info-circle me-2 text-primary"></i>
                <span>Instructions</span>
            </div>
            <div class="card-body">
                <ol class="instructions-list">
                    <li>Enter website URLs (one per line) or upload a text file containing URLs</li>
                    <li>Specify the terms you want to search for, separated by commas</li>
                    <li>Provide your OpenAI API key (never share your API key publicly)</li>
                    <li>Click "Check Websites" to process the list</li>
                    <li>View results in the table below:</li>
                    <ul>
                        <li>"Contains" columns show "Yes" or "No" for each search term</li>
                        <li>"Decision" column shows "Accepted" if all terms are "No", or "Rejected" if any term is "Yes"</li>
                        <li>"Company Description" provides a brief summary of what the website is about</li>
                    </ul>
                    <li>Click "Download Results as CSV" to save the results to your computer</li>
                </ol>
            </div>
        </div>

        <!-- Results Table (Hidden initially) -->
        <div class="card mt-4 fade-in" id="resultsCard" style="display: none;">
            <div class="card-header d-flex justify-content-between align-items-center">
                <div class="d-flex align-items-center">
                    <i class="fas fa-table me-2 text-primary"></i>
                    <span>Analysis Results</span>
                </div>
                <button id="downloadButton" class="btn btn-success" onclick="downloadResults()">
                    <i class="fas fa-download me-2"></i>Download as CSV
                </button>
            </div>
            <div class="card-body">
                <div class="table-container">
                    <table class="table table-hover results-table">
                        <thead>
                            <tr id="resultsTableHeader">
                                <!-- Table headers will be added dynamically -->
                            </tr>
                        </thead>
                        <tbody id="resultsTableBody">
                            <!-- Table rows will be added dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Main JavaScript -->
    <script>
        // Global variables to store analysis data
        let currentAnalysisId = null;
        let currentResults = [];

        // Function to display results in the table
        function displayResults(results) {
            console.log('Displaying results');
            if (!results || results.length === 0) {
                console.log('No results to display');
                return;
            }
            
            // Get all possible columns
            const allColumns = new Set();
            results.forEach(result => {
                Object.keys(result).forEach(key => allColumns.add(key));
            });
            
            // Filter out specific columns to display first
            const priorityColumns = ['Website', 'Status', 'Company Description', 'Decision'];
            
            // Get search term columns
            const searchTermColumns = Array.from(allColumns)
                .filter(col => col.startsWith('Contains \''))
                .sort();
            
            // Get error message column
            const errorMessageColumn = 'Error Message';
            
            // Combine columns in the desired order
            const columns = [
                ...priorityColumns,
                ...searchTermColumns,
                errorMessageColumn
            ];
            
            // Create table header
            const resultsTableHeader = document.getElementById('resultsTableHeader');
            if (resultsTableHeader) {
                resultsTableHeader.innerHTML = '';
                columns.forEach(column => {
                    const th = document.createElement('th');
                    th.textContent = column;
                    resultsTableHeader.appendChild(th);
                });
            }
            
            // Create table rows
            const resultsTableBody = document.getElementById('resultsTableBody');
            if (resultsTableBody) {
                resultsTableBody.innerHTML = '';
                results.forEach(result => {
                    const tr = document.createElement('tr');
                    
                    // Add decision-based row class
                    if (result.Decision === 'Accepted') {
                        tr.classList.add('accepted');
                    } else if (result.Decision === 'Rejected') {
                        tr.classList.add('rejected');
                    }
                    
                    columns.forEach(column => {
                        const td = document.createElement('td');
                        
                        // Format specific columns
                        if (column === 'Decision') {
                            const badge = document.createElement('span');
                            badge.className = 'badge ' + (result[column] === 'Accepted' ? 'badge-success' : 'badge-danger');
                            badge.textContent = result[column] || 'N/A';
                            td.appendChild(badge);
                        } else if (column.startsWith('Contains \'')) {
                            const badge = document.createElement('span');
                            badge.className = 'badge ' + (result[column] === 'Yes' ? 'badge-danger' : 'badge-success');
                            badge.textContent = result[column] || 'N/A';
                            td.appendChild(badge);
                        } else if (column === 'Website') {
                            // Make website a clickable link
                            const link = document.createElement('a');
                            link.href = result[column];
                            link.target = '_blank'; // Open in new tab
                            link.textContent = result[column];
                            link.className = 'text-decoration-none';
                            td.appendChild(link);
                        } else {
                            td.textContent = result[column] || 'N/A';
                        }
                        
                        tr.appendChild(td);
                    });
                    
                    resultsTableBody.appendChild(tr);
                });
            }
        }

        // Function to analyze websites
        function analyzeWebsites() {
            console.log('analyzeWebsites function called');
            
            // Get input values
            const urlsInput = document.getElementById('urlsInput');
            const fileUrlsPreview = document.getElementById('fileUrlsPreview');
            const searchTermsInput = document.getElementById('searchTermsInput');
            const apiKeyInput = document.getElementById('apiKeyInput');
            
            const urls = (urlsInput ? urlsInput.value.trim() : '') || 
                         (fileUrlsPreview ? fileUrlsPreview.value.trim() : '');
            const searchTerms = searchTermsInput ? searchTermsInput.value.trim() : '';
            const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
            
            // Validate inputs
            if (!urls) {
                alert('Please enter at least one URL');
                return;
            }
            
            if (!searchTerms) {
                alert('Please enter at least one search term');
                return;
            }
            
            if (!apiKey) {
                alert('Please enter your OpenAI API key');
                return;
            }
            
            // Show loading overlay using style.visibility
            document.getElementById('loadingOverlay').style.visibility = 'visible';
            
            // Make API request
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    urls: urls,
                    searchTerms: searchTerms,
                    apiKey: apiKey
                }),
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Failed to analyze websites');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Store analysis data
                currentResults = data.results;
                currentAnalysisId = data.analysisId;
                
                // Display results
                displayResults(currentResults);
                
                // Show results card, hide instructions
                document.getElementById('resultsCard').style.display = 'block';
                document.getElementById('instructionsCard').style.display = 'none';
                document.getElementById('resultsCard').scrollIntoView({behavior: 'smooth'});
            })
            .catch(error => {
                console.error('Error analyzing websites:', error);
                alert('Error: ' + error.message);
            })
            .finally(() => {
                // Hide loading overlay using style.visibility
                document.getElementById('loadingOverlay').style.visibility = 'hidden';
            });
        }

        // Function to handle file upload
        function handleFileUpload(event) {
            console.log('File upload handler called');
            const file = event.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const content = e.target.result;
                document.getElementById('fileUrlsPreview').value = content;
                document.getElementById('urlsInput').value = content; // Sync with the URLs input
            };
            reader.readAsText(file);
        }

        // Function to download results as CSV
        function downloadResults() {
            console.log('Downloading results...');
            if (!currentAnalysisId) {
                alert('No analysis results available for download');
                return;
            }
            
            window.location.href = `/api/download/${currentAnalysisId}`;
        }

        // Add event listeners when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM fully loaded');
            
            // File input handler
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                fileInput.addEventListener('change', handleFileUpload);
                console.log('File input event listener added');
            }
            
            // Toggle API key visibility
            const toggleApiKey = document.getElementById('toggleApiKey');
            const apiKeyInput = document.getElementById('apiKeyInput');
            if (toggleApiKey && apiKeyInput) {
                toggleApiKey.addEventListener('click', function() {
                    if (apiKeyInput.type === 'password') {
                        apiKeyInput.type = 'text';
                        toggleApiKey.innerHTML = '<i class="fa fa-eye-slash"></i>';
                    } else {
                        apiKeyInput.type = 'password';
                        toggleApiKey.innerHTML = '<i class="fa fa-eye"></i>';
                    }
                });
                console.log('Toggle API key event listener added');
            }
        });
    </script>
</body>
</html>
"""

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Write the HTML template to a file
with open('templates/index.html', 'w') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 7860))
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)
