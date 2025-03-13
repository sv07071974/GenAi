import gradio as gr
import requests
from bs4 import BeautifulSoup
import openai
import re
import os
import pandas as pd
import concurrent.futures
import time

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

# Function to process multiple websites
def process_multiple_websites(urls_text, search_terms_text, api_key):
    if not urls_text or not search_terms_text:
        return "Please provide both website URLs and text to search for.", None, None

    if not api_key:
        return "Please provide an OpenAI API key.", None, None

    # Parse list of URLs
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

    if not urls:
        return "No valid URLs provided.", None, None

    results = []

    # Process each URL (with option to parallelize)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(check_text_in_website, url, search_terms_text, api_key): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            results.append(result)

    # Create a DataFrame for display
    df = pd.DataFrame(results)

    # Generate a CSV file for download
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"website_analysis_{timestamp}.csv"
    
    # Use a writable directory in Azure (TEMP) or local tmp directory
    temp_dir = os.environ.get('TEMP', '/tmp')
    os.makedirs(temp_dir, exist_ok=True)
    csv_path = os.path.join(temp_dir, csv_filename)
    
    df.to_csv(csv_path, index=False)

    return f"Processing complete. Results saved to {csv_filename}", df, csv_path

# Function to upload URLs from file
def upload_urls_file(file):
    if file is None:
        return ""

    try:
        content = file.decode('utf-8')
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Multi-Website Text Analyzer") as app:
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
            return filename
        return None

    download_button.click(
        fn=prepare_download,
        inputs=[csv_file],
        outputs=[download_output]
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
    6. Click "Download Results as CSV" to save the results to your computer
    """)

# Launch the app
if __name__ == "__main__":
    # Get port from Azure environment or use default
    port = int(os.environ.get("PORT", os.environ.get("WEBSITES_PORT", 8000)))
    
    # Launch with Azure-compatible settings
    app.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=port,
        share=False,  # Don't use Gradio's share feature in production
        debug=False   # Disable debug in production
    )
