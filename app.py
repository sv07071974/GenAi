import gradio as gr
import requests
from bs4 import BeautifulSoup
import openai
import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.error(f"Error fetching website {url}: {str(e)}")
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
        logger.error(f"Translation error: {str(e)}")
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
        logger.error(f"Description generation error: {str(e)}")
        return f"Could not generate description: {str(e)}"

# Function to check if any of the comma-separated terms exist in website content
def check_text_in_website(url, search_terms_text, api_key):
    try:
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
    except Exception as e:
        logger.error(f"Error checking website {url}: {str(e)}")
        return {"Website": url, "Status": "Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": f"Unexpected error: {str(e)}"}

# Function to process multiple websites
def process_multiple_websites(urls_text, search_terms_text, api_key):
    try:
        if not urls_text or not search_terms_text:
            return "Please provide both website URLs and text to search for.", None, None

        if not api_key:
            return "Please provide an OpenAI API key.", None, None

        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

        if not urls:
            return "No valid URLs provided.", None, None

        results = []
        
        # Process sequentially for better stability
        for url in urls:
            result = check_text_in_website(url, search_terms_text, api_key)
            results.append(result)

        df = pd.DataFrame(results)
        return "Processing complete. Results displayed in table below.", df, None
    except Exception as e:
        error_msg = f"Error processing websites: {str(e)}"
        logger.error(error_msg)
        return error_msg, None, None

# Function to upload URLs from file
def upload_urls_file(file):
    if file is None:
        return ""

    try:
        content = file.decode('utf-8')
        return content
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return f"Error reading file: {str(e)}"

# Debug info function
def debug_info():
    info = {
        "current_dir": os.getcwd(),
        "files": os.listdir(),
        "python_path": os.environ.get("PYTHONPATH", "Not set")
    }
    return str(info)

# Create Gradio interface
with gr.Blocks(title="Multi-Website Text Analyzer") as demo:
    gr.Markdown("# Multi-Website Text Analyzer")
    gr.Markdown("This app checks multiple websites to see if they contain specific text.")

    # Store CSV filename for download
    csv_file = gr.State(None)

    with gr.Tab("Enter URLs"):
        with gr.Row():
            with gr.Column():
                urls_input = gr.Textbox(
                    label="Website URLs (one per line)", 
                    placeholder="example.com\nanother-site.org", 
                    lines=8,
                    info="URLs will automatically have 'https://' added if missing and '/' added at the end"
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

    with gr.Tab("Debug"):
        debug_button = gr.Button("Get Debug Info")
        debug_output = gr.Textbox(label="Debug Information", lines=10)
        
        debug_button.click(fn=debug_info, inputs=None, outputs=debug_output)

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

    gr.Markdown("## Instructions")
    gr.Markdown("""
    1. Enter website URLs (one per line) or upload a text file containing URLs
       - You don't need to include 'https://' or trailing slashes - they'll be added automatically
    2. Specify the terms you want to search for, separated by commas
    3. Provide your OpenAI API key (never share your API key publicly)
    4. Click "Check Websites" to process the list
    5. View results in the table:
       - "Contains" columns show "Yes" or "No" for each search term
       - "Decision" column shows "Accepted" if all terms are "No", or "Rejected" if any term is "Yes"
       - "Company Description" provides a brief summary of what the website is about
    """)

# IMPORTANT: Export the ASGI app for Gunicorn
app = demo.app

# For local development only
if __name__ == "__main__":
    demo.launch(share=True)
