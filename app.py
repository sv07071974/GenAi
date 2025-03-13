import os
import time
import concurrent.futures
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
from flask import Flask, request, jsonify, render_template_string, send_file

# Create Flask app
app = Flask(__name__)

# Function to format URL properly
def format_url(url):
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    if not url.endswith('/'):
        url = url + '/'
    return url

# Function to extract text from website
def extract_text_from_url(url):
    try:
        formatted_url = format_url(url)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(formatted_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator='\n')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to translate text to English using OpenAI
def translate_text(text, api_key):
    if not api_key:
        return "Please provide an OpenAI API key", text
        
    try:
        client = openai.OpenAI(api_key=api_key)
        max_chunk_size = 4000
        
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
        words = description.split()
        if len(words) > 20:
            description = " ".join(words[:15]) + "..."
            
        return description
    except Exception as e:
        return f"Could not generate description: {str(e)}"

# Function to check if any of the comma-separated terms exist in website content
def check_text_in_website(url, search_terms_text, api_key):
    formatted_url = format_url(url)
    extracted_text = extract_text_from_url(url)
    
    if isinstance(extracted_text, str) and extracted_text.startswith("Error"):
        return {"Website": formatted_url, "Status": "Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": extracted_text}
        
    translation_status, translated_text = translate_text(extracted_text, api_key)
    
    if translation_status.startswith("Translation error"):
        return {"Website": formatted_url, "Status": "Translation Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": translation_status}
        
    company_description = generate_company_description(translated_text, formatted_url, api_key)
    search_terms = [term.strip().lower() for term in search_terms_text.split(',') if term.strip()]
    
    term_results = {}
    any_term_found = False
    
    for term in search_terms:
        contains_term = "Yes" if term.lower() in translated_text.lower() else "No"
        term_results[term] = contains_term
        
        if contains_term == "Yes":
            any_term_found = True
            
    decision = "Rejected" if any_term_found else "Accepted"
    
    result = {
        "Website": formatted_url,
        "Status": "Success",
        "Company Description": company_description,
        "Decision": decision,
        "Error Message": ""
    }
    
    for term, contains in term_results.items():
        result[f"Contains '{term}'"] = contains
        
    return result

# Function to process multiple websites
def process_multiple_websites(urls_text, search_terms_text, api_key):
    if not urls_text or not search_terms_text:
        return "Please provide both website URLs and text to search for.", []
        
    if not api_key:
        return "Please provide an OpenAI API key.", []
        
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    if not urls:
        return "No valid URLs provided.", []
        
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(check_text_in_website, url, search_terms_text, api_key): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            results.append(result)
            
    df = pd.DataFrame(results)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"website_analysis_{timestamp}.csv"
    
    temp_dir = os.environ.get('TEMP', '/tmp')
    os.makedirs(temp_dir, exist_ok=True)
    csv_path = os.path.join(temp_dir, csv_filename)
    
    df.to_csv(csv_path, index=False)
    
    return f"Processing complete. Results saved to {csv_filename}", results, csv_path

# Basic HTML template for the app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Website Text Analyzer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
        }
        .container {
            margin-top: 20px;
        }
        textarea, input {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 5px solid #ccc;
        }
        .tabs {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            margin-bottom: 20px;
        }
        .tab-button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
        }
        .tab-button:hover {
            background-color: #ddd;
        }
        .tab-button.active {
            background-color: #ccc;
        }
        .tab-content {
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
        }
    </style>
</head>
<body>
    <h1>Multi-Website Text Analyzer</h1>
    <p>This app checks multiple websites to see if they contain specific text.</p>
    
    <div class="tabs">
        <button class="tab-button active" onclick="openTab(event, 'enterUrls')">Enter URLs</button>
        <button class="tab-button" onclick="openTab(event, 'uploadFile')">Upload URLs File</button>
    </div>
    
    <div id="enterUrls" class="tab-content" style="display: block;">
        <form action="/analyze" method="post">
            <div class="container">
                <label for="urls"><b>Website URLs (one per line)</b></label>
                <textarea id="urls" name="urls" rows="8" placeholder="example.com&#10;another-site.org"></textarea>
                
                <label for="search_terms"><b>Terms to Search For (comma-separated)</b></label>
                <input type="text" id="search_terms" name="search_terms" placeholder="term1, term2, term3">
                
                <label for="api_key"><b>OpenAI API Key</b></label>
                <input type="password" id="api_key" name="api_key" placeholder="sk-...">
                
                <button type="submit">Check Websites</button>
            </div>
        </form>
    </div>
    
    <div id="uploadFile" class="tab-content">
        <form action="/upload" method="post" enctype="multipart/form-data">
            <div class="container">
                <label for="file"><b>Upload a text file with URLs (one per line)</b></label>
                <input type="file" id="file" name="file" accept=".txt">
                
                <label for="file_search_terms"><b>Terms to Search For (comma-separated)</b></label>
                <input type="text" id="file_search_terms" name="search_terms" placeholder="term1, term2, term3">
                
                <label for="file_api_key"><b>OpenAI API Key</b></label>
                <input type="password" id="file_api_key" name="api_key" placeholder="sk-...">
                
                <button type="submit">Check Websites from File</button>
            </div>
        </form>
    </div>
    
    {% if status %}
    <div class="status">
        <p><strong>Status:</strong> {{ status }}</p>
        {% if csv_file %}
        <p><a href="/download/{{ csv_file }}" download>Download Results as CSV</a></p>
        {% endif %}
    </div>
    {% endif %}
    
    {% if results %}
    <h2>Results</h2>
    <table>
        <thead>
            <tr>
                <th>Website</th>
                <th>Status</th>
                <th>Company Description</th>
                <th>Decision</th>
                {% for term in terms %}
                <th>Contains '{{ term }}'</th>
                {% endfor %}
                <th>Error Message</th>
            </tr>
        </thead>
        <tbody>
            {% for result in results %}
            <tr>
                <td>{{ result.Website }}</td>
                <td>{{ result.Status }}</td>
                <td>{{ result["Company Description"] }}</td>
                <td>{{ result.Decision }}</td>
                {% for term in terms %}
                <td>{{ result["Contains '" + term + "'"] if "Contains '" + term + "'" in result else "N/A" }}</td>
                {% endfor %}
                <td>{{ result["Error Message"] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
    
    <div class="container">
        <h2>Instructions</h2>
        <ol>
            <li>Enter website URLs (one per line) or upload a text file containing URLs
                <ul>
                    <li>You don't need to include 'https://' or trailing slashes - they'll be added automatically</li>
                </ul>
            </li>
            <li>Specify the terms you want to search for, separated by commas</li>
            <li>Provide your OpenAI API key (never share your API key publicly)</li>
            <li>Click "Check Websites" to process the list</li>
            <li>View results in the table:
                <ul>
                    <li>"Contains" columns show "Yes" or "No" for each search term</li>
                    <li>"Decision" column shows "Accepted" if all terms are "No", or "Rejected" if any term is "Yes"</li>
                    <li>"Company Description" provides a brief summary of what the website is about</li>
                </ul>
            </li>
            <li>Click "Download Results as CSV" to save the results to your computer</li>
        </ol>
    </div>
    
    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tabbuttons;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>
"""

# Routes
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/analyze", methods=["POST"])
def analyze():
    urls_text = request.form.get("urls", "")
    search_terms_text = request.form.get("search_terms", "")
    api_key = request.form.get("api_key", "")
    
    status, results, csv_path = process_multiple_websites(urls_text, search_terms_text, api_key)
    
    terms = [term.strip() for term in search_terms_text.split(',') if term.strip()]
    
    # Get just the filename, not the full path
    csv_file = os.path.basename(csv_path) if csv_path else None
    
    return render_template_string(
        HTML_TEMPLATE, 
        status=status, 
        results=results, 
        terms=terms,
        csv_file=csv_file
    )

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, status="No file part")
        
    file = request.files['file']
    
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, status="No selected file")
        
    if file:
        try:
            urls_text = file.read().decode('utf-8')
            search_terms_text = request.form.get("search_terms", "")
            api_key = request.form.get("api_key", "")
            
            status, results, csv_path = process_multiple_websites(urls_text, search_terms_text, api_key)
            
            terms = [term.strip() for term in search_terms_text.split(',') if term.strip()]
            
            # Get just the filename, not the full path
            csv_file = os.path.basename(csv_path) if csv_path else None
            
            return render_template_string(
                HTML_TEMPLATE, 
                status=status, 
                results=results, 
                terms=terms,
                csv_file=csv_file
            )
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, status=f"Error processing file: {str(e)}")
    
    return render_template_string(HTML_TEMPLATE, status="Unknown error")

@app.route("/download/<filename>")
def download_file(filename):
    temp_dir = os.environ.get('TEMP', '/tmp')
    return send_file(os.path.join(temp_dir, filename), as_attachment=True)

# Health check endpoints for Azure
@app.route('/robots.txt')
@app.route('/robots933456.txt')
def robots():
    return "User-agent: *\nDisallow: /\n"

@app.route('/health')
def health():
    return "OK"

# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("WEBSITES_PORT", 8000)))
    app.run(host='0.0.0.0', port=port, debug=False)
