from flask import Flask, request, jsonify, render_template_string
import requests
from bs4 import BeautifulSoup
import openai
import pandas as pd
import json

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
        response = requests.get(formatted_url, headers=headers, timeout=5)
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

# Website analyzer API endpoint
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        url = data.get('url')
        search_terms = data.get('terms')
        
        if not url or not search_terms:
            return jsonify({"error": "Missing URL or search terms"}), 400
            
        extracted_text = extract_text_from_url(url)
        
        # Simple term checking without OpenAI
        terms_list = [term.strip().lower() for term in search_terms.split(',')]
        results = {}
        
        for term in terms_list:
            results[term] = "Yes" if term.lower() in extracted_text.lower() else "No"
            
        return jsonify({
            "website": url,
            "text_length": len(extracted_text),
            "terms_found": results,
            "sample": extracted_text[:200] + "..."
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# HTML front-end
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Website Analyzer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #4a4a4a; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input, textarea { width: 100%; padding: 8px; }
            button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
            #results { margin-top: 20px; background-color: #f8f9fa; padding: 15px; display: none; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Website Analyzer</h1>
            
            <div class="form-group">
                <label for="url">Website URL:</label>
                <input type="text" id="url" placeholder="example.com">
            </div>
            
            <div class="form-group">
                <label for="terms">Terms to Search For (comma-separated):</label>
                <input type="text" id="terms" placeholder="term1, term2, term3">
            </div>
            
            <button onclick="analyzeWebsite()">Analyze Website</button>
            
            <div id="results">
                <h2>Results</h2>
                <div id="loading" style="display: none;">Analyzing website... This may take a moment.</div>
                <div id="result-content"></div>
            </div>
        </div>
        
        <script>
            function analyzeWebsite() {
                const url = document.getElementById('url').value;
                const terms = document.getElementById('terms').value;
                
                if (!url || !terms) {
                    alert('Please enter both URL and search terms');
                    return;
                }
                
                // Show loading
                document.getElementById('results').style.display = 'block';
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result-content').innerHTML = '';
                
                // Call API
                fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        url: url,
                        terms: terms
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    
                    if (data.error) {
                        document.getElementById('result-content').innerHTML = `<p class="error">Error: ${data.error}</p>`;
                        return;
                    }
                    
                    // Create results table
                    let resultsHtml = `
                        <p><strong>Website:</strong> ${data.website}</p>
                        <p><strong>Text Length:</strong> ${data.text_length} characters</p>
                        <p><strong>Sample Content:</strong> ${data.sample}</p>
                        <h3>Terms Found:</h3>
                        <table>
                            <tr>
                                <th>Term</th>
                                <th>Found</th>
                            </tr>
                    `;
                    
                    for (const [term, found] of Object.entries(data.terms_found)) {
                        resultsHtml += `
                            <tr>
                                <td>${term}</td>
                                <td>${found}</td>
                            </tr>
                        `;
                    }
                    
                    resultsHtml += '</table>';
                    document.getElementById('result-content').innerHTML = resultsHtml;
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('result-content').innerHTML = `<p class="error">Error: ${error.message}</p>`;
                });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
