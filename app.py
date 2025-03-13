# app.py - Fixed for Azure App Service deployment
import gradio as gr
import requests
from bs4 import BeautifulSoup
import openai
import re
import os
import pandas as pd
import concurrent.futures
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Print debug information about the environment
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Environment variables: PORT={os.environ.get('PORT')}, WEBSITES_PORT={os.environ.get('WEBSITES_PORT')}")

# Create necessary directories
os.makedirs('uploads', exist_ok=True)

# Function implementations (same as before)
def format_url(url):
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    if not url.endswith('/'):
        url = url + '/'
    return url

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
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return f"Error: {str(e)}"

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
                        {"role": "system", "content": "Translate to English while preserving formatting."},
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
                    {"role": "system", "content": "Translate to English while preserving formatting."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            translated_text = response.choices[0].message.content
            
        return "Translation successful", translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}", text

def generate_company_description(text, url, api_key):
    if not api_key:
        return "API key required for company description"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        context = text[:3000]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst. Give a concise 15-word description."},
                {"role": "user", "content": f"Website: {url}\nContent: {context}\nDescribe in 15 words:"}
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
        logger.error(f"Could not generate description: {str(e)}")
        return f"Could not generate description: {str(e)}"

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

def process_multiple_websites(urls_text, search_terms_text, api_key):
    if not urls_text or not search_terms_text:
        return "Please provide both website URLs and text to search for.", None, None
    
    if not api_key:
        return "Please provide an OpenAI API key.", None, None
    
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    if not urls:
        return "No valid URLs provided.", None, None
    
    results = []
    
    # Process each URL (with reduced concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_url = {executor.submit(check_text_in_website, url, search_terms_text, api_key): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            results.append(result)
    
    df = pd.DataFrame(results)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"website_analysis_{timestamp}.csv"
    csv_path = os.path.join('uploads', csv_filename)
    df.to_csv(csv_path, index=False)
    
    return f"Processing complete. Results saved to {csv_filename}", df, csv_filename

def upload_urls_file(file):
    if file is None:
        return ""
    try:
        content = file.decode('utf-8')
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Create the Gradio interface
def create_app():
    with gr.Blocks(title="Multi-Website Text Analyzer", analytics_enabled=False) as demo:
        gr.Markdown("# Multi-Website Text Analyzer")
        gr.Markdown("This app checks multiple websites to see if they contain specific text.")
        
        csv_file = gr.State(None)
        
        with gr.Tab("Enter URLs"):
            with gr.Row():
                with gr.Column():
                    urls_input = gr.Textbox(
                        label="Website URLs (one per line)", 
                        placeholder="example.com\nanother-site.org", 
                        lines=8
                    )
                    search_text_input = gr.Textbox(
                        label="Terms to Search For (comma-separated)", 
                        placeholder="term1, term2, term3"
                    )
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key", 
                        placeholder="sk-...", 
                        type="password"
                    )
                    submit_button = gr.Button("Check Websites")
        
        with gr.Tab("Upload URLs File"):
            with gr.Row():
                file_upload = gr.File(label="Upload a text file with URLs (one per line)")
                load_button = gr.Button("Load URLs from File")
            
            with gr.Row():
                file_search_text = gr.Textbox(
                    label="Terms to Search For (comma-separated)", 
                    placeholder="term1, term2, term3"
                )
                file_api_key = gr.Textbox(
                    label="OpenAI API Key", 
                    placeholder="sk-...", 
                    type="password"
                )
                file_submit_button = gr.Button("Check Websites from File")
        
        with gr.Row():
            status_output = gr.Textbox(label="Status")
        
        with gr.Row():
            results_output = gr.DataFrame(label="Results")
        
        with gr.Row():
            download_button = gr.Button("Download Results as CSV")
            download_output = gr.File(label="Download")
        
        # Set up event handlers
        submit_button.click(
            fn=process_multiple_websites,
            inputs=[urls_input, search_text_input, api_key_input],
            outputs=[status_output, results_output, csv_file]
        )
        
        load_button.click(
            fn=upload_urls_file,
            inputs=[file_upload],
            outputs=[urls_input]
        )
        
        file_submit_button.click(
            fn=process_multiple_websites,
            inputs=[urls_input, file_search_text, file_api_key],
            outputs=[status_output, results_output, csv_file]
        )
        
        # Handle download button
        def prepare_download(filename):
            if filename:
                filepath = os.path.join('uploads', filename)
                if os.path.exists(filepath):
                    return filepath
            return None
        
        download_button.click(
            fn=prepare_download,
            inputs=[csv_file],
            outputs=[download_output]
        )
        
        gr.Markdown("## Instructions")
        gr.Markdown("""
        1. Enter website URLs (one per line) or upload a text file containing URLs
        2. Specify the terms you want to search for, separated by commas
        3. Provide your OpenAI API key (never share your API key publicly)
        4. Click "Check Websites" to process the list
        """)

    return demo

# Simple health check HTML response
def get_health_check_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Website Analyzer - Health Check</title>
    </head>
    <body>
        <h1>Website Analyzer</h1>
        <p>Service is running. Please visit the main application at the root URL.</p>
        <p>Current time: {}</p>
    </body>
    </html>
    """.format(time.strftime("%Y-%m-%d %H:%M:%S"))

# Create and launch the app
app = create_app()

# For Azure App Service deployment
if __name__ == "__main__":
    # Determine the port from environment variables - crucial for Azure
    # Azure App Service will set WEBSITES_PORT environment variable
    port = int(os.environ.get("WEBSITES_PORT") or os.environ.get("PORT") or 8000)
    
    logger.info(f"Starting Gradio application on port {port}")
    
    # Custom HTTP health check for Azure
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn
    
    # Create a FastAPI app - this will handle health checks
    fastapi_app = FastAPI()
    
    # Add health check endpoint specifically for Azure
    @fastapi_app.get("/")
    @fastapi_app.get("/health")
    def health():
        return HTMLResponse(content=get_health_check_html())
    
    # Mount the Gradio app at the root
    app.mount_gradio_app(fastapi_app, "/", gradio_api=True)
    
    # Start the FastAPI app on the correct port
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
