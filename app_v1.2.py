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
import shutil
import markdown

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

    # Check if the text indicates CAPTCHA or access issues    
    if text.startswith("Access blocked:") or text.startswith("Error fetching website:") or text.startswith("Error:"):
        return f"Unable to generate description: Access restricted"

    try:
        client = openai.OpenAI(api_key=api_key)

        # Use first 3000 characters of text to get context about company
        context = text[:3000]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst assistant. Based on the website content provided, give a concise 15-word description of what the company or website is about. If you cannot determine what the company is about due to limited information, simply state 'Insufficient information available to determine company purpose.' Never mention CAPTCHA, bot detection, or access limitations in your analysis - focus only on the business information you can find."},
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
# Function to generate company description using OpenAI
def extract_business_details(text, soup, html_content, url, api_key):
    if not api_key:
        return {
            "Products_Services": "API key required",
            "Target_Customers": "API key required",
            "Target_Industries": "API key required",
            "Business_Model": "API key required",
            "Global_Presence": "API key required"
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
        pricing_links = []  # Add specific links for business model info
        customer_links = []  # Add specific links for customer info
        
        # Categorize links by type - expanded to cover more potential source pages
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
            
            # Expanded categorization with more specific types of pages
            if any(term in href or term in link_text for term in ['about', 'company', 'who we are', 'our business']):
                about_links.append((formatted_href, link_text))
            elif any(term in href or term in link_text for term in ['product', 'service', 'solution', 'offering']):
                product_links.append((formatted_href, link_text))
            elif any(term in href or term in link_text for term in ['contact', 'location', 'office', 'global']):
                contact_links.append((formatted_href, link_text))
            elif any(term in href or term in link_text for term in ['pricing', 'plans', 'subscription', 'buy', 'purchase', 'payment']):
                pricing_links.append((formatted_href, link_text))
            elif any(term in href or term in link_text for term in ['customer', 'client', 'success story', 'case study', 'testimonial']):
                customer_links.append((formatted_href, link_text))

        # Dictionary to store source information with links
        sources_info = {
            "Products_Services": {"text": "Main website content", "url": url},
            "Target_Customers": {"text": "Main website content", "url": url},
            "Target_Industries": {"text": "Main website content", "url": url},
            "Business_Model": {"text": "Main website content", "url": url},
            "Global_Presence": {"text": "Main website content", "url": url}
        }
        
        # Get additional context - expanded to include more sources
        additional_context = ""
        
        # About pages for general info
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
                    sources_info["Global_Presence"] = {"text": f"About page: {link_text}", "url": about_url}
                    sources_info["Business_Model"] = {"text": f"About page: {link_text}", "url": about_url}
            except:
                pass
                
        # Product pages for product and industry info
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
                    sources_info["Products_Services"] = {"text": f"Product page: {link_text}", "url": product_url}
                    sources_info["Target_Industries"] = {"text": f"Product page: {link_text}", "url": product_url}
            except:
                pass
                
        # Contact pages for global presence
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
                    sources_info["Global_Presence"] = {"text": f"Contact page: {link_text}", "url": contact_url}
            except:
                pass
        
        # Pricing pages for business model details
        for pricing_url, link_text in pricing_links[:1]:  # Limit to first pricing page
            try:
                pricing_result, _, _ = extract_text_from_url(pricing_url)
                if isinstance(pricing_result, str) and not pricing_result.startswith("Error"):
                    context_sample = pricing_result[:1500]
                    additional_context += f"\nContent from {pricing_url} ({link_text}):\n{context_sample}"
                    
                    # Store full content for raw export
                    all_pages_content[f"pricing_page_{len(all_pages_content)}"] = {
                        "url": pricing_url,
                        "title": f"Pricing Page: {link_text}",
                        "content": pricing_result[:20000]  # Limit to 20k chars
                    }
                    
                    # Update sources for business model
                    sources_info["Business_Model"] = {"text": f"Pricing page: {link_text}", "url": pricing_url}
            except:
                pass
                
        # Customer pages for target customer details
        for customer_url, link_text in customer_links[:1]:  # Limit to first customer page
            try:
                customer_result, _, _ = extract_text_from_url(customer_url)
                if isinstance(customer_result, str) and not customer_result.startswith("Error"):
                    context_sample = customer_result[:1500]
                    additional_context += f"\nContent from {customer_url} ({link_text}):\n{context_sample}"
                    
                    # Store full content for raw export
                    all_pages_content[f"customer_page_{len(all_pages_content)}"] = {
                        "url": customer_url,
                        "title": f"Customer Page: {link_text}",
                        "content": customer_result[:20000]  # Limit to 20k chars
                    }
                    
                    # Update sources for target customers
                    sources_info["Target_Customers"] = {"text": f"Customer page: {link_text}", "url": customer_url}
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
        
        # Create prompt for GPT with optimized instructions
        prompt = f"""
You are a business intelligence expert analyzing {company_name}'s website. Provide extremely detailed information about the company in these five areas:

1. Products & Services: What products/services does the company offer? Include all relevant categories, types, features, and details.

2. Target Customers: Who are their customers? Include all customer segments, sizes (SMB, enterprise), demographics, use cases, and any named clients or testimonials.

3. Target Industries: What industries/sectors does the company serve? List all industry verticals mentioned and any specialized industry solutions.

4. Business Model: How does the company make money? Include revenue model (subscription, licensing, transaction fees), pricing structure, sales channels, partnerships, and go-to-market approach.

5. Global Presence & Group Structure: Where does the company operate? Include headquarters, office locations, international presence, parent company relationships, subsidiaries, and organizational structure.

EXTREMELY IMPORTANT FORMAT REQUIREMENTS:
* Write in complete paragraphs with detailed information
* DO NOT use category labels, prefixes or transitions like "The company offers" or "The business model is"
* Start directly with the facts - NO category names or descriptive intros
* Maintain ALL detail and specific information from the content
* For each section, add "Source: [URL]" on a new line at the end

Website: {url}
Meta information: {meta_info}
Content:
{combined_context}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a business intelligence analyst who extracts detailed, factual information and presents it without unnecessary labels or transitions. Your analysis is comprehensive and direct."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,  # Increased for more comprehensive information
            temperature=0.2   # Lower temperature for more focused response
        )

        # Parse the response into categories
        raw_response = response.choices[0].message.content.strip()
        
        # Initialize default structure
        business_details = {
            "Products_Services": "Information not available",
            "Products_Services_Source": f"{url} (Main website)",
            
            "Target_Customers": "Information not available",
            "Target_Customers_Source": f"{url} (Main website)",
            
            "Target_Industries": "Information not available",
            "Target_Industries_Source": f"{url} (Main website)",
            
            "Business_Model": "Information not available",
            "Business_Model_Source": f"{url} (Main website)",
            
            "Global_Presence": "Information not available",
            "Global_Presence_Source": f"{url} (Main website)"
        }
        
        # Store raw content for validation
        business_details["_raw_content_files"] = all_pages_content
        
        # Split the response by numbered sections
        section_splits = re.split(r'\n\s*\d+\.\s+', '\n' + raw_response)
        if len(section_splits) >= 6:  # First element is empty due to the split pattern
            # Map each section to its corresponding category
            categories = ["Products_Services", "Target_Customers", "Target_Industries", "Business_Model", "Global_Presence"]
            
            for i, category in enumerate(categories):
                content = section_splits[i+1].strip()  # +1 because first split is empty
                
                # Remove any remaining category labels
                content = re.sub(r'^(Products & Services|Products and Services|Target Customers|Customer Segments|Target Industries|Industries Served|Business Model|Revenue Model|Global Presence|Group Structure)[\s:]*', '', content, flags=re.IGNORECASE)
                
                # Extract source if mentioned in the content
                source_match = re.search(r'Source: (.+?)(?:$|\n)', content)
                if source_match:
                    source_text = source_match.group(1).strip()
                    # Clean the content by removing the source mention
                    content = re.sub(r'Source: .+?($|\n)', '', content).strip()
                    
                    # If a specific page was mentioned as source
                    found_url = None
                    for page_key, page_data in all_pages_content.items():
                        if page_key != "main_page" and source_text.lower() in page_data.get("title", "").lower():
                            found_url = page_data["url"]
                            break
                            
                    if found_url:
                        sources_info[category] = {"text": source_text, "url": found_url}
                
                # Store the clean content
                if content and content != "Information not available":
                    business_details[category] = content
                
                # Add source information with URL
                source_text = sources_info[category]["text"]
                source_url = sources_info[category]["url"]
                business_details[f"{category}_Source"] = f"{source_url} ({source_text})"
        
        return business_details
    except Exception as e:
        error_msg = f"Error extracting data: {str(e)}"
        return {
            "Products_Services": error_msg,
            "Target_Customers": "Error extracting data",
            "Target_Industries": "Error extracting data",
            "Business_Model": "Error extracting data",
            "Global_Presence": "Error extracting data"
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

def generate_text_reports(results):
    if not results:
        return "No results to generate reports from."
    
    # Create a directory for text reports
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    directory = f"reports_{timestamp}"
    try:
        os.makedirs(directory, exist_ok=True)
    except:
        # If directory creation fails, use current directory
        directory = ""
    
    # Generate a text file for each website
    report_files = []
    
    for result in results:
        website_url = result.get("Website", "unknown")
        
        # Skip if there's no website URL
        if website_url == "unknown":
            continue
        
        # Create a safe filename from the website URL
        clean_url = website_url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
        if len(clean_url) > 30:
            clean_url = clean_url[:30]
        
        # Generate report content
        report_content = f"BUSINESS ANALYSIS FOR {website_url}\n"
        report_content += "=" * 80 + "\n\n"
        
        # Add company description
        company_desc = result.get("Company Description", "Not available")
        report_content += f"COMPANY DESCRIPTION\n{'-' * 20}\n{company_desc}\n\n"
        
        # Add detected terms
        report_content += f"SEARCH TERM DETECTION\n{'-' * 20}\n"
        for key, value in result.items():
            if key.startswith("Contains '") and key.endswith("'"):
                term = key[len("Contains '"):-1]
                report_content += f"- {term}: {value}\n"
        report_content += "\n"
        
        # Add business details with sources
        categories = [
            ("PRODUCTS & SERVICES", "Products_Services", "Products_Services_Source"),
            ("TARGET CUSTOMERS", "Target_Customers", "Target_Customers_Source"),
            ("TARGET INDUSTRIES", "Target_Industries", "Target_Industries_Source"),
            ("BUSINESS MODEL", "Business_Model", "Business_Model_Source"),
            ("GLOBAL PRESENCE", "Global_Presence", "Global_Presence_Source")
        ]
        
        report_content += f"BUSINESS DETAILS\n{'-' * 20}\n\n"
        
        for display_name, key, source_key in categories:
            value = result.get(key, "Information not available")
            source = result.get(source_key, "")
            
            # Clean format with clear section headers
            report_content += f"{display_name}\n{'-' * len(display_name)}\n"
            report_content += f"{value}\n\n"
            if source:
                report_content += f"Source: {source}\n\n"
        
        # Write to text file
        txt_filename = f"{directory}/{clean_url}_report.txt"
        try:
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            report_files.append(txt_filename)
        except Exception as e:
            print(f"Error writing text file {txt_filename}: {str(e)}")
    
    # Create a zip file with all report files
    zip_filename = f"{directory}/all_reports.zip"
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in report_files:
                zipf.write(file)
        
        return zip_filename
    except Exception as e:
        print(f"Error creating zip file: {str(e)}")
        # Return the individual files if zip creation fails
        return report_files[0] if report_files else None

# Function to generate raw content files for all websites
def generate_all_raw_content(raw_website_data_dict):
    if not raw_website_data_dict:
        return None
        
    # Create a directory for the files
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    directory = f"raw_content_{timestamp}"
    try:
        os.makedirs(directory, exist_ok=True)
    except:
        # If directory creation fails, use current directory
        directory = ""
    
    # Process each website
    all_filenames = []
    
    for website_url, raw_content_data in raw_website_data_dict.items():
        # Create subdirectory for each website
        clean_url = website_url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
        if len(clean_url) > 30:
            clean_url = clean_url[:30]
            
        website_dir = f"{directory}/{clean_url}"
        try:
            os.makedirs(website_dir, exist_ok=True)
        except:
            continue
        
        # Create files for each page
        for page_key, page_data in raw_content_data.items():
            page_url = page_data.get("url", "unknown")
            page_title = page_data.get("title", page_key)
            content = page_data.get("content", "No content available")
            
            # Create filename
            if page_key == "main_page":
                filename = f"{website_dir}/main.txt"
            else:
                # Create a safe filename from the page title
                safe_title = "".join(c if c.isalnum() else "_" for c in page_title)
                if len(safe_title) > 20:
                    safe_title = safe_title[:20]
                filename = f"{website_dir}/{safe_title}.txt"
            
            # Write content to file
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"URL: {page_url}\n")
                    f.write(f"Title: {page_title}\n")
                    f.write("="*50 + "\n\n")
                    f.write(content)
                
                all_filenames.append(filename)
            except Exception as e:
                print(f"Error writing file {filename}: {str(e)}")
    
    # Create a zip file with all content files
    zip_filename = f"{directory}/all_raw_content.zip"
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in all_filenames:
                zipf.write(file)
        
        return zip_filename
    except Exception as e:
        print(f"Error creating zip file: {str(e)}")
        return None

# Dictionary to store all results and raw website data
all_results = []
raw_website_data_dict = {}

# Function to process multiple websites
def process_multiple_websites(urls_text, search_terms_text, api_key):
    global all_results, raw_website_data_dict
    
    if not urls_text or not search_terms_text:
        return "Please provide both website URLs and text to search for.", None, None

    if not api_key:
        return "Please provide an OpenAI API key.", None, None

    # Parse list of URLs
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

    if not urls:
        return "No valid URLs provided.", None, None

    results = []
    raw_website_data_dict = {}  # Reset global data

    # Process each URL (with option to parallelize)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(check_text_in_website, url, search_terms_text, api_key): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            
            # Extract and store raw content data for download
            if "_raw_content_files" in result:
                raw_content = result.pop("_raw_content_files")
                raw_website_data_dict[result["Website"]] = raw_content
            
            results.append(result)
    
    # Store results globally
    all_results = results
    
    # Create a DataFrame for display
    df = pd.DataFrame(results)

    # Generate a CSV file for download - include everything in the CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"website_analysis_{timestamp}.csv"
    pd.DataFrame(results).to_csv(csv_filename, index=False)

    return f"Processing complete. Results saved to {csv_filename}", df, csv_filename

# Function to upload URLs from file
def upload_urls_file(file):
    if file is None:
        return ""

    try:
        content = file.decode('utf-8')
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Function to create markdown reports download
def create_markdown_reports():
    global all_results
    return generate_text_reports(all_results)

# Function to create raw content download for all websites
def create_all_raw_content():
    global raw_website_data_dict
    return generate_all_raw_content(raw_website_data_dict)

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
        
    with gr.Row():
        with gr.Column(scale=1):
            download_button = gr.Button("Download Results as CSV")
            download_output = gr.File(label="Download CSV")
        
        with gr.Column(scale=1):
            download_markdown_button = gr.Button("Download All Markdown Reports")
            markdown_output = gr.File(label="Download Markdown")
            
        with gr.Column(scale=1):
            download_raw_button = gr.Button("Download All Raw Content")
            raw_content_output = gr.File(label="Download Raw Content")

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
    
    download_button.click(
        fn=lambda f: f,
        inputs=[csv_file],
        outputs=[download_output]
    )
    
    download_markdown_button.click(
        fn=create_markdown_reports,
        inputs=[],
        outputs=[markdown_output]
    )
    
    download_raw_button.click(
        fn=create_all_raw_content,
        inputs=[],
        outputs=[raw_content_output]
    )
    
    gr.Markdown("""
    ## Instructions
    
    1. Enter website URLs (one per line) or upload a text file containing URLs
       - You don't need to include 'https://' or trailing slashes - they'll be added automatically
    2. Specify the terms you want to search for, separated by commas
    3. Provide your OpenAI API key (never share your API key publicly)
    4. Click "Check Websites" to process the list
    5. View results in the table
    6. Download options:
       - Click "Download Results as CSV" to save all analysis results to your computer
       - Click "Download All Markdown Reports" to get organized markdown files for each website
       - Click "Download All Raw Content" to get the original content from all analyzed websites
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080)
