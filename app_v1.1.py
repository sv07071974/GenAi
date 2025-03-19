import gradio as gr
import requests
from bs4 import BeautifulSoup
import openai
import re
import os
import pandas as pd
import concurrent.futures
import time
import zipfile

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

        return text, soup, response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching website: {str(e)}", None, None
    except Exception as e:
        return f"Error: {str(e)}", None, None

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

# Function to extract detailed business information using OpenAI
def extract_business_details(text, soup, html_content, url, api_key):
    if not api_key:
        return {
            "Products & Services": "API key required",
            "Target Customers": "API key required",
            "Target Industries": "API key required",
            "Business Model": "API key required",
            "Global Presence & Group Structure": "API key required"
        }

    try:
        client = openai.OpenAI(api_key=api_key)

        # Dictionary to store all pages content for raw export
        all_pages_content = {
            "main_page": {
                "url": url,
                "title": "Main Page",
                "content": text[:20000]  # Limit to 20k chars for main page
            }
        }
        
        # First identify the most relevant sources
        sources = {}
        
        # Check for common pages that might contain business information
        about_links = []
        product_links = []
        contact_links = []
        
        # Categorize links by type
        for link in soup.find_all('a', href=True):
            href = link.get('href').lower()
            link_text = link.get_text().lower().strip()
            if not link_text:
                continue  # Skip empty links
            
            # Format URL properly
            formatted_href = href
            if href.startswith('/'):
                formatted_href = url + href.lstrip('/')
            elif not href.startswith(('http://', 'https://')):
                formatted_href = url + href
            
            # Categorize link
            if any(term in href or term in link_text for term in ['about', 'company', 'who we are', 'our business']):
                about_links.append((formatted_href, link_text))
            elif any(term in href or term in link_text for term in ['product', 'service', 'solution', 'offering']):
                product_links.append((formatted_href, link_text))
            elif any(term in href or term in link_text for term in ['contact', 'location', 'office', 'global']):
                contact_links.append((formatted_href, link_text))

        # Dictionary to store source information with links
        sources_info = {
            "Products & Services": {"text": "Main website content", "url": url},
            "Target Customers": {"text": "Main website content", "url": url},
            "Target Industries": {"text": "Main website content", "url": url},
            "Business Model": {"text": "Main website content", "url": url},
            "Global Presence & Group Structure": {"text": "Main website content", "url": url}
        }
        
        # Get additional context from about or company pages (if available)
        additional_context = ""
        for about_url, link_text in about_links[:2]:  # Limit to first 2 about pages
            try:
                about_result, _, _ = extract_text_from_url(about_url)
                if isinstance(about_result, str) and not about_result.startswith("Error"):
                    context_sample = about_result[:1500]
                    additional_context += f"\nContent from {about_url} ({link_text}):\n{context_sample}"
                    
                    # Store full content for raw export
                    all_pages_content[f"about_page_{len(all_pages_content)}"] = {
                        "url": about_url,
                        "title": f"About Page: {link_text}",
                        "content": about_result[:20000]  # Limit to 20k chars
                    }
                    
                    # Update sources for company structure and general info
                    sources_info["Global Presence & Group Structure"] = {"text": f"About page: {link_text}", "url": about_url}
                    sources_info["Business Model"] = {"text": f"About page: {link_text}", "url": about_url}
            except:
                pass
                
        # Get product information
        for product_url, link_text in product_links[:2]:  # Limit to first 2 product pages
            try:
                product_result, _, _ = extract_text_from_url(product_url)
                if isinstance(product_result, str) and not product_result.startswith("Error"):
                    context_sample = product_result[:1500]
                    additional_context += f"\nContent from {product_url} ({link_text}):\n{context_sample}"
                    
                    # Store full content for raw export
                    all_pages_content[f"product_page_{len(all_pages_content)}"] = {
                        "url": product_url,
                        "title": f"Product Page: {link_text}",
                        "content": product_result[:20000]  # Limit to 20k chars
                    }
                    
                    # Update sources for product information
                    sources_info["Products & Services"] = {"text": f"Product page: {link_text}", "url": product_url}
                    sources_info["Target Industries"] = {"text": f"Product page: {link_text}", "url": product_url}
            except:
                pass
                
        # Get location/contact information for global presence
        for contact_url, link_text in contact_links[:1]:  # Limit to first contact page
            try:
                contact_result, _, _ = extract_text_from_url(contact_url)
                if isinstance(contact_result, str) and not contact_result.startswith("Error"):
                    context_sample = contact_result[:1000]
                    additional_context += f"\nContent from {contact_url} ({link_text}):\n{context_sample}"
                    
                    # Store full content for raw export
                    all_pages_content[f"contact_page_{len(all_pages_content)}"] = {
                        "url": contact_url,
                        "title": f"Contact Page: {link_text}",
                        "content": contact_result[:20000]  # Limit to 20k chars
                    }
                    
                    # Update sources for global presence
                    sources_info["Global Presence & Group Structure"] = {"text": f"Contact page: {link_text}", "url": contact_url}
            except:
                pass
        
        # Use first 4000 characters of main text + additional context from about pages
        main_context = text[:4000]
        combined_context = main_context + additional_context
        
        # Parse title and meta descriptions for additional context
        meta_info = ""
        if soup:
            title = soup.find('title')
            if title:
                meta_info += f"Website title: {title.get_text()}\n"
                
            for meta in soup.find_all('meta'):
                if meta.get('name') in ['description', 'keywords']:
                    meta_info += f"Meta {meta.get('name')}: {meta.get('content')}\n"
        
        # Extract company name from URL or title
        company_name = url.split('//')[1].split('.')[0]
        if soup and soup.find('title'):
            title_text = soup.find('title').get_text()
            company_name = title_text.split(' - ')[0].split(' | ')[0]
        
        # Create the prompt for GPT
        prompt = f"""
You are a business intelligence analyst. Extract the following information about {company_name} from their website content:

1. Products & Services: What types of products or services does the company offer? List categories and types.
2. Target Customers: Who are their key customer segments (e.g., businesses, consumers, government)?
3. Target Industries: What sectors or industries does the company serve (e.g., healthcare, retail, automotive)?
4. Business Model: What is their business model (e.g., wholesale, retail, manufacturing, SaaS, service-based)?
5. Global Presence & Group Structure: Do they have a presence in multiple countries? Are they affiliated with a group of companies or parent organization?

For each category, provide a concise summary based ONLY on explicit information from the content provided. If information is unavailable for a category, state "Information not available".

Also, for each category, indicate the exact source URL where this information was found. If information comes from the main website, indicate that.

Website: {url}
Meta information: {meta_info}
Content:
{combined_context}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a business intelligence analyst who extracts precise information from website content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        # Parse the response into categories
        raw_response = response.choices[0].message.content.strip()
        
        # Initialize default structure
        business_details = {
            "Products & Services": "Information not available",
            "Products & Services Source": f"{url} (Main website)",
            
            "Target Customers": "Information not available",
            "Target Customers Source": f"{url} (Main website)",
            
            "Target Industries": "Information not available",
            "Target Industries Source": f"{url} (Main website)",
            
            "Business Model": "Information not available",
            "Business Model Source": f"{url} (Main website)",
            
            "Global Presence & Group Structure": "Information not available",
            "Global Presence & Group Structure Source": f"{url} (Main website)"
        }
        
        # Store raw content for validation
        business_details["_raw_content_files"] = all_pages_content
        
        # Extract each section from the response
        sections = {
            "Products & Services": r"(?:1\.|Products & Services:)(.+?)(?=2\.|Target Customers:|$)",
            "Target Customers": r"(?:2\.|Target Customers:)(.+?)(?=3\.|Target Industries:|$)",
            "Target Industries": r"(?:3\.|Target Industries:)(.+?)(?=4\.|Business Model:|$)",
            "Business Model": r"(?:4\.|Business Model:)(.+?)(?=5\.|Global Presence & Group Structure:|$)",
            "Global Presence & Group Structure": r"(?:5\.|Global Presence & Group Structure:)(.+)(?=$)"
        }
        
        for category, pattern in sections.items():
            match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                # Clean up any bullet points or unnecessary markers
                content = re.sub(r'^\s*[-•*]\s*', '', content, flags=re.MULTILINE)
                
                # Extract source if mentioned in the content
                source_match = re.search(r'Source: (.+?)(?:\.|$)', content)
                if source_match:
                    source_text = source_match.group(1).strip()
                    # Clean the content by removing the source mention
                    content = re.sub(r'Source: .+?(?:\.|$)', '', content).strip()
                    
                    # If a specific page was mentioned as source
                    found_url = None
                    for page_key, page_data in all_pages_content.items():
                        if page_key != "main_page" and source_text.lower() in page_data.get("title", "").lower():
                            found_url = page_data["url"]
                            break
                            
                    if found_url:
                        sources_info[category] = {"text": source_text, "url": found_url}
                        
                business_details[category] = content
                
                # Add source information with URL (plain text version since HTML might not render)
                source_text = sources_info[category]["text"]
                source_url = sources_info[category]["url"]
                business_details[f"{category} Source"] = f"{source_url} ({source_text})"
                
        return business_details
    except Exception as e:
        error_msg = f"Error extracting data: {str(e)}"
        return {
            "Products & Services": error_msg,
            "Target Customers": "Error extracting data",
            "Target Industries": "Error extracting data",
            "Business Model": "Error extracting data",
            "Global Presence & Group Structure": "Error extracting data"
        }

# Function to check if any of the comma-separated terms exist in website content
def check_text_in_website(url, search_terms_text, api_key):
    # Format the URL for display
    formatted_url = format_url(url)

    # Extract text from website
    extracted_result = extract_text_from_url(url)
    
    if isinstance(extracted_result, tuple) and len(extracted_result) == 3:
        extracted_text, soup, html_content = extracted_result
    else:
        extracted_text = extracted_result
        soup = None
        html_content = None

    if isinstance(extracted_text, str) and extracted_text.startswith("Error"):
        return {"Website": formatted_url, "Status": "Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": extracted_text}

    # Translate text to English
    translation_status, translated_text = translate_text(extracted_text, api_key)

    if translation_status.startswith("Translation error"):
        return {"Website": formatted_url, "Status": "Translation Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": translation_status}

    # Generate company description
    company_description = generate_company_description(translated_text, formatted_url, api_key)

    # Extract detailed business information
    business_details = extract_business_details(translated_text, soup, html_content, formatted_url, api_key)

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
    
    # Add business details 
    for key, value in business_details.items():
        # Skip internal raw content storage
        if not key.startswith('_'):
            result[key] = value
    
    # Store raw content separately with a key that won't be displayed
    if '_raw_content_files' in business_details:
        result['_raw_content_files'] = business_details['_raw_content_files']

    return result

# Function to generate raw content files for validation
def generate_raw_content_files(raw_content_data, website_url):
    if not raw_content_data:
        return None
        
    # Create a directory for the files if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    directory = f"raw_content_{timestamp}"
    try:
        os.makedirs(directory, exist_ok=True)
    except:
        # If directory creation fails, use current directory
        directory = ""
    
    # Clean URL for filename
    clean_url = website_url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
    if len(clean_url) > 30:
        clean_url = clean_url[:30]  # Limit filename length
    
    # Create individual files for each page
    filenames = []
    
    for page_key, page_data in raw_content_data.items():
        page_url = page_data.get("url", "unknown")
        page_title = page_data.get("title", page_key)
        content = page_data.get("content", "No content available")
        
        # Create filename
        if page_key == "main_page":
            filename = f"{directory}/{clean_url}_main.txt"
        else:
            # Create a safe filename from the page title
            safe_title = "".join(c if c.isalnum() else "_" for c in page_title)
            if len(safe_title) > 20:
                safe_title = safe_title[:20]
            filename = f"{directory}/{clean_url}_{safe_title}.txt"
        
        # Write content to file
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"URL: {page_url}\n")
                f.write(f"Title: {page_title}\n")
                f.write("="*50 + "\n\n")
                f.write(content)
            
            filenames.append(filename)
        except Exception as e:
            print(f"Error writing file {filename}: {str(e)}")
    
    # Create a zip file with all content files
    zip_filename = f"{directory}/{clean_url}_all_content.zip"
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in filenames:
                zipf.write(file)
        
        return zip_filename
    except Exception as e:
        print(f"Error creating zip file: {str(e)}")
        # Return the individual files if zip creation fails
        return filenames[0] if filenames else None

# Dictionary to store raw website data
raw_website_data_dict = {}
websites_list = []

# Function to update website list display
def update_website_list():
    global websites_list
    if not websites_list:
        return "No websites analyzed yet. Run the analysis first."
    return "\n".join(websites_list)

# Function to process multiple websites
def process_multiple_websites(urls_text, search_terms_text, api_key):
    global raw_website_data_dict, websites_list
    
    if not urls_text or not search_terms_text:
        return "Please provide both website URLs and text to search for.", None, None, "No websites analyzed yet."

    if not api_key:
        return "Please provide an OpenAI API key.", None, None, "No websites analyzed yet."

    # Parse list of URLs
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

    if not urls:
        return "No valid URLs provided.", None, None, "No websites analyzed yet."

    results = []
    raw_website_data_dict = {}  # Reset global data
    websites_list = []   # Reset list

    # Process each URL (with option to parallelize)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(check_text_in_website, url, search_terms_text, api_key): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            
            # Extract and store raw content data for download
            if "_raw_content_files" in result:
                raw_content = result.pop("_raw_content_files")
                raw_website_data_dict[result["Website"]] = raw_content
                websites_list.append(result["Website"])  # Add to list
            
            results.append(result)
    
    # Create a DataFrame for display
    df = pd.DataFrame(results)

    # Generate a CSV file for download - include everything in the CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"website_analysis_{timestamp}.csv"
    pd.DataFrame(results).to_csv(csv_filename, index=False)
    
    # Return the website list as a string
    websites_str = "Available websites for raw content download:\n" + "\n".join(websites_list)

    return f"Processing complete. Results saved to {csv_filename}", df, csv_filename, websites_str

# Function to upload URLs from file
def upload_urls_file(file):
    if file is None:
        return ""

    try:
        content = file.decode('utf-8')
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Function to create raw content download for a specific website
def create_raw_content_download(website_url):
    global raw_website_data_dict
    
    if not website_url or website_url.strip() == "":
        return None
    
    # Format the URL properly
    website_url = format_url(website_url.strip())
    
    if website_url not in raw_website_data_dict:
        return None
    
    website_data = raw_website_data_dict[website_url]
    try:
        return generate_raw_content_files(website_data, website_url)
    except Exception as e:
        print(f"Error generating raw content: {str(e)}")
        return None

# Create Gradio interface
with gr.Blocks(title="Multi-Website Text Analyzer") as app:
    gr.Markdown("# Multi-Website Text Analyzer")
    gr.Markdown("This app checks multiple websites to see if they contain specific text and extracts business details with source links.")

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
    
    # Add website selector for raw content
    with gr.Row():
        websites_display = gr.Textbox(
            label="Available Websites for Raw Content Download", 
            placeholder="Run analysis to see available websites",
            lines=5,
            interactive=False
        )
        
    with gr.Row():
        selected_website = gr.Textbox(
            label="Enter Website URL for Raw Content Download", 
            placeholder="Copy and paste a URL from the list above",
            interactive=True
        )
        
    with gr.Row():
        col1, col2 = gr.Column(scale=1), gr.Column(scale=1)
        
        with col1:
            download_button = gr.Button("Download Results as CSV")
            download_output = gr.File(label="Download CSV")
        
        with col2:
            download_raw_button = gr.Button("Download Raw Content")
            raw_content_output = gr.File(label="Raw Content Download")

    # Set up event handlers
    submit_button.click(
        fn=process_multiple_websites,
        inputs=[urls_input, search_text_input, api_key_input],
        outputs=[status_output, results_output, csv_file, websites_display]
    )

    load_button.click(
        fn=upload_urls_file,
        inputs=[file_upload],
        outputs=[urls_input]
    )

    file_submit_button.click(
        fn=process_multiple_websites,
        inputs=[urls_input, file_search_text, file_api_key],
        outputs=[status_output, results_output, csv_file, websites_display]
    )
    
    download_button.click(
        fn=lambda f: f,
        inputs=[csv_file],
        outputs=[download_output]
    )
    
    download_raw_button.click(
        fn=create_raw_content_download,
        inputs=[selected_website],
        outputs=[raw_content_output]
    )
    
    gr.Markdown("""
    ## Instructions
    
    1. Enter website URLs (one per line) or upload a text file containing URLs
       - You don't need to include 'https://' or trailing slashes - they'll be added automatically
    2. Specify the terms you want to search for, separated by commas
    3. Provide your OpenAI API key (never share your API key publicly)
    4. Click "Check Websites" to process the list
    5. View results in the table:
       - "Contains" columns show "Yes" or "No" for each search term
       - "Decision" column shows "Accepted" if all terms are "No", or "Rejected" if any term is "Yes"
       - "Company Description" provides a brief summary of what the website is about
       - Detailed business information is provided with source links
    6. Download options:
       - Click "Download Results as CSV" to save all analysis results to your computer
       - Copy a website URL from the list, paste it in the input field, and click "Download Raw Content"
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080)
