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
import hashlib
import markdown

# Cache for API responses to avoid duplicate calls
api_cache = {}

# Function to format URL properly
def format_url(url):
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    if not url.endswith('/'):
        url = url + '/'
    return url

# Optimize text before sending to API to reduce tokens
def preprocess_text_for_api(text, max_length=3000):
    """Optimize text before sending to OpenAI API"""
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # If text is too long, keep important parts
    if len(text) > max_length:
        # Keep beginning and end as they often contain key information
        begin_text = text[:int(max_length*0.7)]  # 70% from start
        end_text = text[-int(max_length*0.3):]   # 30% from end
        text = begin_text + " [...] " + end_text
    
    return text

# Function to extract text from website with optimizations
def extract_text_from_url(url):
    try:
        # Format the URL before use
        formatted_url = format_url(url)

        # Send request with optimized headers
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            # First try with SSL verification
            response = requests.get(formatted_url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            # Fallback without verification
            response = requests.get(formatted_url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Parse HTML content more efficiently
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, and navigation elements
        for element in soup(["script", "style", "nav", "footer", "iframe"]):
            element.extract()

        # Get text more efficiently
        text = soup.get_text(separator='\n')
        
        # Clean text (remove extra whitespace and empty lines)
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)

        return text, soup, response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching website: {str(e)}", None, None
    except Exception as e:
        return f"Error: {str(e)}", None, None

# Function to check if text is likely English to avoid unnecessary translation
def is_likely_english(text, sample_size=1000):
    # Common English words for a simple check
    common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 
                       'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 
                       'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 
                       'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 
                       'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 
                       'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 
                       'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 
                       'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 
                       'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'])
    
    # Take a sample of the text for efficiency
    sample = text[:sample_size].lower()
    
    # Count words that appear in common English words
    words = re.findall(r'\b\w+\b', sample)
    if not words:
        return True  # Default to English if no words found
        
    english_count = sum(1 for word in words if word in common_words)
    
    # If more than 15% of words are common English words, likely English
    return (english_count / len(words) > 0.15) if words else True

# Cached API call function to avoid duplicate processing
def get_cached_or_call_api(text, prompt, api_key, model="gpt-3.5-turbo", max_tokens=300, temperature=0.3, system_prompt="You are a business analyst who provides concise, structured analysis."):
    """Cache API results to avoid duplicate calls"""
    # Create a cache key from text and prompt
    cache_key = hashlib.md5((text + prompt).encode()).hexdigest()
    
    # Return cached result if available
    if cache_key in api_cache:
        return api_cache[cache_key]
    
    # Otherwise call API
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        result = response.choices[0].message.content
        
        # Cache the result
        api_cache[cache_key] = result
        return result
    except Exception as e:
        return f"API Error: {str(e)}"

# Function to translate text to English using OpenAI (optimized)
def translate_text(text, api_key):
    if not api_key:
        return "Please provide an OpenAI API key", text

    # Skip translation if already English
    if is_likely_english(text):
        return "No translation needed (text appears to be in English)", text

    try:
        # Process in smaller chunks for efficiency
        max_chunk_size = 3000  # Reduced from 4000
        if len(text) > max_chunk_size:
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []

            for chunk in chunks:
                prompt = f"Translate the following text to English while preserving formatting:\n\n{chunk}"
                result = get_cached_or_call_api(
                    chunk, 
                    prompt, 
                    api_key, 
                    max_tokens=3500, 
                    temperature=0.3,
                    system_prompt="You are a helpful translation assistant."
                )
                translated_chunks.append(result)

            translated_text = "".join(translated_chunks)
        else:
            prompt = f"Translate the following text to English while preserving formatting:\n\n{text}"
            translated_text = get_cached_or_call_api(
                text, 
                prompt, 
                api_key, 
                max_tokens=3500, 
                temperature=0.3,
                system_prompt="You are a helpful translation assistant."
            )

        return "Translation successful", translated_text
    except Exception as e:
        return f"Translation error: {str(e)}", text

# Function to generate company description using OpenAI (optimized)
def generate_company_description(text, url, api_key):
    if not api_key:
        return "API key required for company description"

    # Check for access issues    
    if isinstance(text, str) and (text.startswith("Access blocked:") or text.startswith("Error fetching website:") or text.startswith("Error:")):
        return f"Unable to generate description: Access restricted"

    try:
        # Use only first 2000 characters (reduced from 3000)
        context = preprocess_text_for_api(text[:2000], 2000)
        
        prompt = f"Website: {url}\n\nContent sample: {context}\n\nProvide a concise 15-word description of what this company or website is about:"
        
        description = get_cached_or_call_api(
            context,
            prompt,
            api_key,
            max_tokens=50,
            temperature=0.7,
            system_prompt="You are a business analyst assistant. Based on the website content provided, give a concise description of what the company or website is about. If you cannot determine what the company is about due to limited information, simply state 'Insufficient information available to determine company purpose.'"
        )

        # Ensure description is around 15 words
        words = description.split()
        if len(words) > 20:  # Allow some flexibility
            description = " ".join(words[:15]) + "..."

        return description
    except Exception as e:
        return f"Could not generate description: {str(e)}"

# Combined function to analyze business nature and extract countries
def analyze_business_nature_and_countries(text, url, api_key):
    """Combine multiple API calls into one"""
    if not api_key:
        return {
            "Business_Nature": "API key required",
            "Nature_of_Customers": "API key required",
            "Nature_of_Operations": "API key required",
            "Countries_of_Operation": "API key required"
        }

    # Check for access issues    
    if isinstance(text, str) and (text.startswith("Access blocked:") or text.startswith("Error fetching website:") or text.startswith("Error:")):
        return {
            "Business_Nature": "Unable to analyze: Access restricted",
            "Nature_of_Customers": "Unable to analyze: Access restricted",
            "Nature_of_Operations": "Unable to analyze: Access restricted",
            "Countries_of_Operation": "Unable to analyze: Access restricted"
        }

    try:
        # Optimize text before sending
        context = preprocess_text_for_api(text[:3500], 3500)
        
        # One prompt that gets all information
        prompt = f"""
Based on this website content, provide ONLY these 4 data points:
1. Nature of Customers: ONLY "Business to Business" or "Business to Consumer" or "Both"
2. Nature of Operations: ONLY "Manufacturing", "Trading", "Services", or specific combinations like "Manufacturing, Trading"
3. Countries of Operation: comma-separated list of countries/regions mentioned (say "Not available" if none found)
4. Business Nature: what industry or business sector they operate in and what specific products/services they offer (1-2 sentences)

Website: {url}
Content: {context}

Format your response EXACTLY as follows (no additional text):
Nature of Customers: [Your answer]
Nature of Operations: [Your answer]
Countries of Operation: [Your answer]
Business Nature: [Your answer]
"""
        
        result = get_cached_or_call_api(
            context,
            prompt,
            api_key,
            max_tokens=350,
            temperature=0.2,
            system_prompt="You are a business analyst who analyzes companies based on their website content. Provide direct, factual answers based only on the available evidence."
        )
        
        # Parse the response into components
        business_analysis = {
            "Business_Nature": "Unable to determine",
            "Nature_of_Customers": "Unable to determine",
            "Nature_of_Operations": "Unable to determine",
            "Countries_of_Operation": "Not available"
        }
        
        # Extract each component
        customer_match = re.search(r'Nature of Customers:\s*(.*?)(?:\n|$)', result)
        if customer_match:
            business_analysis["Nature_of_Customers"] = customer_match.group(1).strip()
            
        operation_match = re.search(r'Nature of Operations:\s*(.*?)(?:\n|$)', result)
        if operation_match:
            business_analysis["Nature_of_Operations"] = operation_match.group(1).strip()
            
        countries_match = re.search(r'Countries of Operation:\s*(.*?)(?:\n|$)', result)
        if countries_match:
            business_analysis["Countries_of_Operation"] = countries_match.group(1).strip()
            
        nature_match = re.search(r'Business Nature:\s*(.*?)(?:\n|$)', result)
        if nature_match:
            business_analysis["Business_Nature"] = nature_match.group(1).strip()
        
        return business_analysis
    except Exception as e:
        return {
            "Business_Nature": f"Error: {str(e)}",
            "Nature_of_Customers": "Error during analysis",
            "Nature_of_Operations": "Error during analysis",
            "Countries_of_Operation": "Error during analysis"
        }

# Optimized function to extract business details
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
        # Dictionary to store all pages content for raw export
        all_pages_content = {
            "main_page": {
                "url": url,
                "title": "Main Page",
                "content": text[:20000]  # Limit to 20k chars for main page
            }
        }
        
        # First identify the most relevant sources - focusing only on key pages
        sources = {}
        about_links = []
        product_links = []
        
        # Categorize links by type - only focusing on most critical page types
        if soup:
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
                
                # Optimized categorization - only about and product pages
                if any(term in href or term in link_text for term in ['about', 'company', 'who we are', 'our business']):
                    about_links.append((formatted_href, link_text))
                elif any(term in href or term in link_text for term in ['product', 'service', 'solution', 'offering']):
                    product_links.append((formatted_href, link_text))
        
        # Dictionary to store source information with links
        sources_info = {
            "Products_Services": {"text": "Main website content", "url": url},
            "Target_Customers": {"text": "Main website content", "url": url},
            "Target_Industries": {"text": "Main website content", "url": url},
            "Business_Model": {"text": "Main website content", "url": url},
            "Global_Presence": {"text": "Main website content", "url": url}
        }
        
        # Only fetch 1 about page and 1 product page for efficiency
        additional_context = ""
        
        # About pages for general info
        for about_url, link_text in about_links[:1]:  # Reduced to just 1 about page
            try:
                about_result, _, _ = extract_text_from_url(about_url)
                if isinstance(about_result, str) and not about_result.startswith("Error"):
                    # Take only first 1500 chars instead of full page
                    context_sample = preprocess_text_for_api(about_result[:1500], 1500)
                    additional_context += f"\nContent from About page:\n{context_sample}"
                    
                    # Store content for raw export
                    all_pages_content[f"about_page_{len(all_pages_content)}"] = {
                        "url": about_url,
                        "title": f"About Page: {link_text}",
                        "content": about_result[:20000]
                    }
                    
                    # Update sources
                    sources_info["Global_Presence"] = {"text": f"About page", "url": about_url}
                    sources_info["Business_Model"] = {"text": f"About page", "url": about_url}
            except:
                pass
                
        # Product pages for product and industry info
        for product_url, link_text in product_links[:1]:  # Reduced to just 1 product page
            try:
                product_result, _, _ = extract_text_from_url(product_url)
                if isinstance(product_result, str) and not product_result.startswith("Error"):
                    # Take only first 1500 chars instead of full page
                    context_sample = preprocess_text_for_api(product_result[:1500], 1500)
                    additional_context += f"\nContent from Product page:\n{context_sample}"
                    
                    # Store content for raw export
                    all_pages_content[f"product_page_{len(all_pages_content)}"] = {
                        "url": product_url,
                        "title": f"Product Page: {link_text}",
                        "content": product_result[:20000]
                    }
                    
                    # Update sources
                    sources_info["Products_Services"] = {"text": f"Product page", "url": product_url}
                    sources_info["Target_Industries"] = {"text": f"Product page", "url": product_url}
            except:
                pass
        
        # Use first 3000 characters of main text + additional context
        main_context = preprocess_text_for_api(text[:3000], 3000)
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
        
        # More efficient prompt
        prompt = f"""
Analyze {company_name}'s website. Provide detailed information in these areas:

1. Products & Services: What products/services do they offer?
2. Target Customers: Who are their customers?
3. Target Industries: What industries do they serve?
4. Business Model: How do they make money?
5. Global Presence: Where do they operate?

FORMAT: For each section, write paragraphs with detailed information. Do NOT use category labels or transitions. Start directly with facts. For each section, add "Source: [URL]" on a new line at the end.

Website: {url}
Meta information: {meta_info}
Content:
{combined_context}
"""

        result = get_cached_or_call_api(
            combined_context,
            prompt,
            api_key,
            model="gpt-3.5-turbo-16k",
            max_tokens=1500,
            temperature=0.2,
            system_prompt="You are a business intelligence analyst who extracts detailed, factual information and presents it without unnecessary labels or transitions. Your analysis is comprehensive and direct."
        )
        
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
        section_splits = re.split(r'\n\s*\d+\.\s+', '\n' + result)
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

# Modified function to check website content with optimized processing
def check_text_in_website(url, business_terms_text, api_key, customer_types=None, operation_types=None):
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

    # Translate text to English if needed
    translation_status, translated_text = translate_text(extracted_text, api_key)

    if translation_status.startswith("Translation error"):
        return {"Website": formatted_url, "Status": "Translation Error", "Company Description": "N/A", "Decision": "N/A", "Error Message": translation_status}

    # Generate company description
    company_description = generate_company_description(translated_text, formatted_url, api_key)

    # Extract detailed business information
    business_details = extract_business_details(translated_text, soup, html_content, formatted_url, api_key)
    
    # Combined analysis of business nature and countries (optimized - single API call)
    combined_analysis = analyze_business_nature_and_countries(translated_text, formatted_url, api_key)
    
    business_analysis = {
        "Business_Nature": combined_analysis.get("Business_Nature", "Unable to determine"),
        "Nature_of_Customers": combined_analysis.get("Nature_of_Customers", "Unable to determine"),
        "Nature_of_Operations": combined_analysis.get("Nature_of_Operations", "Unable to determine")
    }
    
    countries = combined_analysis.get("Countries_of_Operation", "Not available")

    # Parse comma-separated business terms
    business_terms = [term.strip().lower() for term in business_terms_text.split(',') if term.strip()]

    # Initialize result dictionary for each term
    term_results = {}
    any_term_found = False

    for term in business_terms:
        # Check if term is in translated content (case insensitive)
        contains_term = "Yes" if term.lower() in translated_text.lower() else "No"
        term_results[term] = contains_term

        if contains_term == "Yes":
            any_term_found = True

    # Initialize all conditions as False by default
    customer_type_match = False
    operation_type_match = False
    business_terms_match = False
    
    # Check customer type match
    if customer_types and len(customer_types) > 0:
        if business_analysis["Nature_of_Customers"] == "Unable to determine":
            customer_type_match = False  # If unable to determine, consider it a non-match
        else:
            # Check if the business customer type matches any of the selected types
            if business_analysis["Nature_of_Customers"] == "Both":
                customer_type_match = any(ct in ["Both", "Business to Business", "Business to Consumer"] for ct in customer_types)
            else:
                customer_type_match = business_analysis["Nature_of_Customers"] in customer_types
    else:
        customer_type_match = True  # If no customer types are specified, consider it a match
    
    # Check operation type match
    if operation_types and len(operation_types) > 0:
        if business_analysis["Nature_of_Operations"] == "Unable to determine":
            operation_type_match = False  # If unable to determine, consider it a non-match
        else:
            # Parse the business operation types
            business_operation_types = [op.strip() for op in business_analysis["Nature_of_Operations"].split(',')]
            
            # Check if any of the business operation types match any of the selected types
            operation_type_match = False
            for business_op in business_operation_types:
                if any(op == business_op for op in operation_types):
                    operation_type_match = True
                    break
    else:
        operation_type_match = True  # If no operation types are specified, consider it a match
    
    # Check if any of the specified business terms are found
    if business_terms:
        business_terms_match = any_term_found
    else:
        business_terms_match = True  # If no business terms are specified, consider it a match
    
    # All conditions must be True for acceptance
    if customer_type_match and operation_type_match and business_terms_match:
        decision = "Accepted"
    else:
        # Provide detailed rejection reason
        rejection_reasons = []
        if not customer_type_match:
            rejection_reasons.append("Customer Type Mismatch")
        if not operation_type_match:
            rejection_reasons.append("Operation Type Mismatch")
        if not business_terms_match:
            rejection_reasons.append("No Business Terms Found")
        
        decision = f"Rejected ({', '.join(rejection_reasons)})"

    # Add basic information to result
    result = {
        "Website": formatted_url,
        "Status": "Success",
        "Company Description": company_description,
        "Business Nature": business_analysis["Business_Nature"],
        "Nature of Customers": business_analysis["Nature_of_Customers"],
        "Nature of Operations": business_analysis["Nature_of_Operations"],
        "Countries of Operation": countries,
        "Decision": decision,
        "Error Message": ""
    }

    # Add result for each business term
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

# Function to upload URLs file
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

# Function to process multiple websites with optimized processing
# Continuation of process_multiple_websites function
def process_multiple_websites(urls_text, business_terms_text, api_key, customer_types=None, operation_types=None):
    global all_results, raw_website_data_dict
    
    if not urls_text:
        return "Please provide website URLs to analyze.", None, None

    if not api_key:
        return "Please provide an OpenAI API key.", None, None

    # Parse list of URLs
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

    if not urls:
        return "No valid URLs provided.", None, None

    results = []
    raw_website_data_dict = {}  # Reset global data

    # Create a mapping from original URL to index to maintain order
    url_to_index = {url: idx for idx, url in enumerate(urls)}
    
    # Process websites in smaller batches for efficiency
    batch_size = 5
    url_batches = [urls[i:i+batch_size] for i in range(0, len(urls), batch_size)]
    indexed_results = []
    
    for batch in url_batches:
        # Process each batch with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # Reduced max_workers
            future_to_url = {executor.submit(check_text_in_website, url, business_terms_text, api_key, customer_types, operation_types): url for url in batch}
            
            # Collect results from this batch
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                result = future.result()
                
                # Extract and store raw content data for download
                if "_raw_content_files" in result:
                    raw_content = result.pop("_raw_content_files")
                    raw_website_data_dict[result["Website"]] = raw_content
                
                # Clean up any leading hyphens in all text fields
                for key, value in result.items():
                    if isinstance(value, str) and value.startswith("- "):
                        result[key] = value[2:]  # Remove leading "- "
                    elif isinstance(value, str) and value.startswith("-"):
                        result[key] = value[1:].lstrip()  # Remove leading "-" and any spaces
                
                # Store result with original index
                indexed_results.append((url_to_index[url], result))
    
    # Sort results back to original order
    indexed_results.sort(key=lambda x: x[0])
    results = [result for _, result in indexed_results]
    
    # Store results globally
    all_results = results
    
    # Create a DataFrame for display
    df = pd.DataFrame(results)

    # Generate a CSV file for download - include everything in the CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"website_analysis_{timestamp}.csv"
    
    # Clean data before saving to CSV to prevent Excel formula issues
    csv_df = df.copy()
    
    # Function to clean strings to prevent Excel formula issues
    def clean_for_excel(value):
        if isinstance(value, str):
            # Remove any leading hyphens with spaces
            if value.startswith("- "):
                value = value[2:]
            elif value.startswith("-"):
                value = value[1:].lstrip()
            
            # Add a single quote prefix to prevent Excel from interpreting as formula
            if value.startswith(('=', '+', '-', '@')):
                return "'" + value
            # Also escape any value that looks like it might cause issues
            if any(char in value for char in ['#', '=']):
                return "'" + value
        return value
    
    # Apply cleaning to all string values
    for col in csv_df.columns:
        csv_df[col] = csv_df[col].apply(clean_for_excel)
    
    # Save to CSV with proper escaping
    csv_df.to_csv(csv_filename, index=False, escapechar='\\', quoting=1)

    return f"Processing complete. Results saved to {csv_filename}", df, csv_filename

# Report generation functions (optimized to maintain functionality)
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
        
        # Add business type analysis
        report_content += f"BUSINESS TYPE ANALYSIS\n{'-' * 20}\n"
        report_content += f"Business Nature: {result.get('Business Nature', 'Not available')}\n"
        report_content += f"Nature of Customers: {result.get('Nature of Customers', 'Not available')}\n"
        report_content += f"Nature of Operations: {result.get('Nature of Operations', 'Not available')}\n"
        report_content += f"Countries of Operation: {result.get('Countries of Operation', 'Not available')}\n\n"
        
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

# User interface function - remains the same but using optimized backend functions
def create_ui():
    # Dictionary to store all results and raw website data
    all_results = []
    raw_website_data_dict = {}
    
    # Create Gradio interface with the same UI from original code
    with gr.Blocks(
        title="Multi-Website Text Analyzer V2",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray",
            font=["Aptos", "sans-serif"]
        ),
        css="""
        /* CSS styles remain the same as original */
        /* Coolors Palette: https://coolors.co/palette/495867-577399-bdd5ea-f7f7ff-fe5f55 */
        /* 495867 - Charcoal - Dark text, headings */
        /* 577399 - UCLA Blue - Secondary elements */
        /* BDD5EA - Light Steel Blue - Borders, separators */
        /* F7F7FF - Ghost White - Background */
        /* FE5F55 - Orange Red Crayola - Accent color */
        
        /* Import Aptos font */
        @import url('https://fonts.googleapis.com/css2?family=Aptos:wght@300;400;500;600;700&display=swap');
        
        body, input, button, select, textarea, label {
            font-family: 'Aptos', sans-serif !important;
        }
        
        body {
            background-color: #F7F7FF !important;
        }
        
        .title-container {
            text-align: center;
            margin-bottom: 1.5rem;
            padding: 1.5rem 0;
            border-bottom: 2px solid #BDD5EA;
            background: linear-gradient(to right, rgba(247, 247, 255, 0), rgba(254, 95, 85, 0.1), rgba(247, 247, 255, 0));
        }
        
        .title {
            font-family: 'Aptos', sans-serif;
            font-size: 2.5rem;
            font-weight: 600;
            color: #495867;
            margin-bottom: 0.25rem;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            font-family: 'Aptos', sans-serif;
            font-size: 1.2rem;
            font-weight: 300;
            color: #577399;
            font-style: italic;
        }
        
        .section-header {
            font-family: 'Aptos', sans-serif;
            font-weight: 500;
            color: #495867;
            font-size: 1.3rem;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid #BDD5EA;
            padding-bottom: 0.5rem;
        }
        
        .card {
            border: 1px solid #BDD5EA;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(73, 88, 103, 0.05);
            background: #F7F7FF;
            margin-bottom: 1rem;
        }
        
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            font-style: italic;
            color: #577399;
            font-size: 0.9rem;
            border-top: 1px solid #BDD5EA;
            font-family: 'Aptos', sans-serif;
            font-weight: 300;
        }
        
        button.primary {
            background-color: #FE5F55 !important;
            border: none !important;
            color: #F7F7FF !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            font-family: 'Aptos', sans-serif !important;
        }
        
        button.primary:hover {
            background-color: #e54c42 !important;
            box-shadow: 0 4px 8px rgba(254, 95, 85, 0.3) !important;
        }
        
        button.secondary {
            background-color: #BDD5EA !important;
            border: 1px solid #577399 !important;
            color: #495867 !important;
            transition: all 0.3s ease !important;
            font-family: 'Aptos', sans-serif !important;
            font-weight: 500 !important;
        }
        
        button.secondary:hover {
            background-color: #a9c5e0 !important;
        }
        
        .instructions-card {
            background-color: #F7F7FF;
            border-left: 4px solid #FE5F55;
        }
        
        /* Additional styling for form elements */
        input, textarea, select {
            background-color: #F7F7FF !important;
            border: 1px solid #BDD5EA !important;
            color: #495867 !important;
            box-shadow: inset 0 1px 2px rgba(73, 88, 103, 0.1) !important;
            font-family: 'Aptos', sans-serif !important;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: #FE5F55 !important;
            box-shadow: 0 0 0 3px rgba(254, 95, 85, 0.2) !important;
            outline: none !important;
        }
        
        /* Tab styling - more professional look */
        .tab-nav {
            background-color: #F0F3F9 !important;
            border-bottom: 2px solid #BDD5EA !important;
            padding-bottom: 0 !important;
        }
        
        .tab-nav button {
            color: #577399 !important;
            font-family: 'Aptos', sans-serif !important;
            font-weight: 500 !important;
            border-bottom: 3px solid transparent !important;
            border-radius: 0 !important;
            margin-bottom: -2px !important;
            position: relative !important;
            background: transparent !important;
            transition: all 0.3s ease !important;
        }
        
        .tab-nav button.selected {
            background-color: transparent !important;
            color: #FE5F55 !important;
            font-weight: 600 !important;
            border-bottom: 3px solid #FE5F55 !important;
        }

        /* Processing indicator */
        .processing-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            margin: 10px 0;
            background-color: rgba(254, 95, 85, 0.1);
            border-radius: 5px;
            border-left: 3px solid #FE5F55;
        }
        
        .processing-indicator-text {
            margin-left: 10px;
            font-style: italic;
            color: #577399;
            font-family: 'Aptos', sans-serif;
            font-weight: 300;
        }
        
        /* Classic spinner animation */
        @keyframes spinner {
            to {transform: rotate(360deg);}
        }
        
        .spinner {
            display: inline-block;
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            border: 4px solid #BDD5EA;
            border-top-color: #FE5F55;
            animation: spinner 1s linear infinite;
        }
        
        /* Enhanced table styling with fixed headers and scrollbars */
        .gradio-dataframe {
            overflow: auto !important;
            max-height: 500px !important;
        }
        
        .gradio-dataframe table {
            width: 100%;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            font-family: 'Aptos', sans-serif !important;
        }
        
        .gradio-dataframe table thead {
            position: sticky !important;
            top: 0 !important;
            z-index: 100 !important;
        }
        
        .gradio-dataframe table th {
            background-color: #495867 !important;
            color: #F7F7FF !important;
            font-weight: 500 !important;
            text-align: left !important;
            padding: 12px !important;
            border-bottom: 2px solid #FE5F55 !important;
            position: sticky !important;
            top: 0 !important;
            font-family: 'Aptos', sans-serif !important;
        }
        
        .gradio-dataframe table td {
            padding: 10px 12px !important;
            border-bottom: 1px solid #BDD5EA !important;
            color: #495867 !important;
            white-space: normal !important;
            word-break: break-word !important;
            font-family: 'Aptos', sans-serif !important;
        }
        
        .gradio-dataframe table tr:nth-child(even) {
            background-color: rgba(189, 213, 234, 0.2) !important;
        }
        
        .gradio-dataframe table tr:hover {
            background-color: rgba(254, 95, 85, 0.05) !important;
        }
        
        /* Custom scrollbar styling */
        .gradio-dataframe::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        .gradio-dataframe::-webkit-scrollbar-track {
            background: #F0F3F9;
            border-radius: 4px;
        }
        
        .gradio-dataframe::-webkit-scrollbar-thumb {
            background: #BDD5EA;
            border-radius: 4px;
        }
        
        .gradio-dataframe::-webkit-scrollbar-thumb:hover {
            background: #577399;
        }
        
        /* Status indicator styling */
        .status-indicator {
            display: flex;
            align-items: center;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #F0F3F9;
            font-family: 'Aptos', sans-serif;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-waiting {
            background-color: #BDD5EA;
        }
        
        .status-processing {
            background-color: #FE5F55;
        }
        
        .status-success {
            background-color: #4CAF50;
        }
        
        .status-error {
            background-color: #F44336;
        }
        
        /* Filter section styling */
        .filter-section {
            padding: 5px 10px;
            background-color: rgba(189, 213, 234, 0.2);
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .filter-title {
            font-weight: 600;
            color: #495867;
            margin-bottom: 5px;
            font-family: 'Aptos', sans-serif;
        }
        
        /* Checkbox group styling */
        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .checkbox-item {
            display: flex;
            align-items: center;
            background-color: #F0F3F9;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .checkbox-item:hover {
            background-color: #BDD5EA;
        }
        
        .checkbox-item input {
            margin-right: 6px;
        }
        
        .checkbox-item label {
            font-family: 'Aptos', sans-serif;
            font-size: 0.9rem;
            color: #495867;
            cursor: pointer;
        }
        """
    ) as app:
        # Custom HTML for title section
        gr.HTML("""
            <div class="title-container">
                <h1 class="title">Multi-Website Text Analyzer V2</h1>
                <p class="subtitle">Analyze business websites and extract valuable insights with elegant precision</p>
                <div style="width: 60px; height: 5px; background-color: #FE5F55; margin: 15px auto;"></div>
            </div>
        """)

        # Store CSV filename for download
        csv_file = gr.State(None)
        
        # Processing indicator (initially hidden)
        with gr.Row(visible=False) as processing_indicator:
            gr.HTML("""<div style="text-align: center; margin-bottom: 20px;">
  <div style="font-family: 'Aptos', sans-serif; font-size: 22px; color: #FE5F55; font-weight: 600;">Processing websites...</div>
</div>
<div class="particle-loader">
  <div class="particle-track"></div>
  <div class="particle" style="--i:1;"></div>
  <div class="particle" style="--i:2;"></div>
  <div class="particle" style="--i:3;"></div>
  <div class="particle" style="--i:4;"></div>
  <div class="particle" style="--i:5;"></div>
</div>
<style>
.particle-loader {
  width: 300px;
  height: 20px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 10px;
  position: relative;
  margin: 50px auto;
  overflow: hidden;
}
.particle-track {
  position: absolute;
  height: 100%;
  width: 60%;
  background: linear-gradient(90deg, #FE5F55, #FF9F97);
  border-radius: 10px;
  animation: trackExpand 3s infinite alternate;
}
.particle {
  position: absolute;
  width: 10px;
  height: 10px;
  background: white;
  border-radius: 50%;
  top: 5px;
  left: calc(60% - 5px);
  animation: particleFloat 2s infinite ease-in-out;
  animation-delay: calc(var(--i) * 0.2s);
  opacity: 0.7;
}
.particle-text {
  position: absolute;
  top: 30px;
  left: 0;
  width: 100%;
  text-align: center;
  font-family: 'Aptos', sans-serif;
  font-size: 18px;
  font-weight: 500;
  color: #FE5F55;
  text-shadow: 0 1px 2px rgba(0,0,0,0.05);
  animation: textPulse 1.5s infinite;
}

@keyframes textPulse {
  0%, 100% { opacity: 0.8; }
  50% { opacity: 1; }
}

@keyframes trackExpand {
  0% { width: 30%; }
  100% { width: 80%; }
}
@keyframes particleFloat {
  0%, 100% {
    transform: translateY(0) translateX(0);
    opacity: 0.7;
  }
  50% {
    transform: translateY(-15px) translateX(10px);
    opacity: 0.3;
  }
}
</style>

<script>
    // Simple auto-increment function that will definitely work
    (function() {
        const progressBar = document.getElementById('simpleProgressBar');
        let width = 10;
        
        function frame() {
            // Increment width
            if (width < 90) {
                width += 0.5;
                progressBar.style.width = width + '%';
            }
            
            // Continue animation immediately
            setTimeout(frame, 100);
        }
        
        // Start animation immediately
        frame();
    })();
</script>
    """)

        with gr.Tabs() as tabs:
            with gr.TabItem("Enter URLs", id=1):
                with gr.Column(variant="card"):
                    gr.HTML('<div class="section-header">Enter Website Information</div>')
                    urls_input = gr.Textbox(
                        label="Website URLs (one per line)", 
                        placeholder="example.com\nanother-site.org", 
                        lines=8,
                        info="URLs will automatically have 'https://' added if missing and '/' added at the end"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            business_terms_input = gr.Textbox(
                                label="Nature of Business (comma-separated)", 
                                placeholder="software, manufacturing, retail",
                                info="Enter terms that describe the business nature you're looking for"
                            )
                    
                    # Place Nature of Customers & Nature of Operations side by side
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="filter-title">Nature of Customers:</div>')
                            customer_type_options = ["Business to Business", "Business to Consumer", "Both"]
                            customer_type_input = gr.CheckboxGroup(
                                choices=customer_type_options,
                                label="",
                                elem_classes=["checkbox-group"]
                            )
                        
                        with gr.Column(scale=1):
                            gr.HTML('<div class="filter-title">Nature of Operations:</div>')
                            operation_type_options = ["Manufacturing", "Trading", "Services"]
                            operation_type_input = gr.CheckboxGroup(
                                choices=operation_type_options,
                                label="",
                                elem_classes=["checkbox-group"]
                            )
                            
                    # Enhanced checkbox styling
                    gr.HTML("""<style>
    /* Modern checkbox styling with high visibility */
    .checkbox-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 10px 0;
    }
    
    /* Individual checkbox item */
    .checkbox-button {
        display: inline-flex !important;
        align-items: center !important;
        background-color: #f0f0f0 !important;
        border: 2px solid #d0d0d0 !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    /* Hover state */
    .checkbox-button:hover {
        background-color: #e8e8e8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Selected state */
    .checkbox-button.selected {
        background-color: #e1f5fe !important;
        border-color: #29b6f6 !important;
        box-shadow: 0 0 0 3px rgba(41, 182, 246, 0.3) !important;}
    
    /* Selected state for checkbox input */
    .checkbox-button.selected input[type="checkbox"] {
        background-color: #29b6f6 !important;
        border-color: #29b6f6 !important;
    }
    
    /* Checkbox input styling */
    .checkbox-button input[type="checkbox"] {
        position: relative !important;
        width: 24px !important;
        height: 24px !important;
        margin-right: 10px !important;
        -webkit-appearance: none !important;
        appearance: none !important;
        background-color: white !important;
        border: 2px solid #757575 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Checkmark */
    .checkbox-button input[type="checkbox"]:checked::after {
        content: "✓" !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        font-size: 16px !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    /* Checkbox label styling */
    .checkbox-button label {
        font-family: 'Aptos', sans-serif !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #333333 !important;
        margin-left: 5px !important;
        cursor: pointer !important;
    }
    
    /* Selected label styling */
    .checkbox-button.selected label {
        color: #0277bd !important;
        font-weight: 600 !important;
    }
    
    /* Original classes for compatibility */
    .checkbox-item {
        display: inline-flex !important;
        align-items: center !important;
        background-color: #f0f0f0 !important;
        border: 2px solid #d0d0d0 !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    .checkbox-item:hover {
        background-color: #e8e8e8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    .checkbox-item input[type="checkbox"] {
        position: relative !important;
        width: 24px !important;
        height: 24px !important;
        margin-right: 10px !important;
        -webkit-appearance: none !important;
        appearance: none !important;
        background-color: white !important;
        border: 2px solid #757575 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .checkbox-item input[type="checkbox"]:checked {
        background-color: #29b6f6 !important;
        border-color: #29b6f6 !important;
    }
    
    .checkbox-item input[type="checkbox"]:checked::after {
        content: "✓" !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        font-size: 16px !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    .checkbox-group {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
        padding: 15px 0 !important;
    }
</style>

<script>
    // Enhance checkbox functionality
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for Gradio to render elements
        setTimeout(() => {
            // Find all checkbox inputs
            const checkboxInputs = document.querySelectorAll('input[type="checkbox"]');
            
            checkboxInputs.forEach(checkbox => {
                // Find the parent label
                const parentLabel = checkbox.closest('label');
                if (parentLabel) {
                    // Add our custom class
                    parentLabel.classList.add('checkbox-button');
                    
                    // Add selected class if checked
                    if (checkbox.checked) {
                        parentLabel.classList.add('selected');
                    }
                    
                    // Add event listener to toggle selected class
                    checkbox.addEventListener('change', () => {
                        if (checkbox.checked) {
                            parentLabel.classList.add('selected');
                        } else {
                            parentLabel.classList.remove('selected');
                        }
                    });
                    
                    // Make the entire button clickable
                    parentLabel.addEventListener('click', (e) => {
                        // Prevent double-triggering for actual checkbox
                        if (e.target !== checkbox) {
                            e.preventDefault();
                            checkbox.checked = !checkbox.checked;
                            
                            // Manually trigger change event
                            const event = new Event('change', { bubbles: true });
                            checkbox.dispatchEvent(event);
                            
                            // Update selected class
                            if (checkbox.checked) {
                                parentLabel.classList.add('selected');
                            } else {
                                parentLabel.classList.remove('selected');
                            }
                        }
                    });
                }
            });
        }, 1000);
        
        // Reapply styling periodically to catch any dynamically added elements
        setInterval(() => {
            const checkboxInputs = document.querySelectorAll('input[type="checkbox"]:not(.styled)');
            checkboxInputs.forEach(checkbox => {
                checkbox.classList.add('styled');
                const parentLabel = checkbox.closest('label');
                if (parentLabel && !parentLabel.classList.contains('checkbox-button')) {
                    parentLabel.classList.add('checkbox-button');
                    if (checkbox.checked) {
                        parentLabel.classList.add('selected');
                    }
                }
            });
        }, 2000);
    });
</script>
""")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            api_key_input = gr.Textbox(
                                label="OpenAI API Key", 
                                placeholder="sk-...", 
                                type="password"
                            )
                    
                    with gr.Row():
                        submit_button = gr.Button("Check Websites", variant="primary", elem_classes=["primary"])

            with gr.TabItem("Upload URLs File", id=2):
                with gr.Column(variant="card"):
                    gr.HTML('<div class="section-header">Upload File with URLs</div>')
                    file_upload = gr.File(label="Upload a text file with URLs (one per line)")
                    load_button = gr.Button("Load URLs from File", elem_classes=["secondary"])

                    with gr.Row():
                        with gr.Column(scale=2):
                            file_business_terms = gr.Textbox(
                                label="Nature of Business (comma-separated)", 
                                placeholder="software, manufacturing, retail"
                            )
                    
                    # Nature of Customers - Checkbox group for file tab
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="filter-title">Nature of Customers:</div>')
                            file_customer_type = gr.CheckboxGroup(
                                choices=customer_type_options,
                                label="",
                                elem_classes=["checkbox-group"]
                            )
                        
                        with gr.Column(scale=1):
                            gr.HTML('<div class="filter-title">Nature of Operations:</div>')
                            file_operation_type = gr.CheckboxGroup(
                                choices=operation_type_options,
                                label="",
                                elem_classes=["checkbox-group"]
                            )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_api_key = gr.Textbox(
                                label="OpenAI API Key", 
                                placeholder="sk-...", 
                                type="password"
                            )
                    
                    with gr.Row():
                        file_submit_button = gr.Button("Check Websites from File", variant="primary", elem_classes=["primary"])

        # Status output with enhanced status indicator
        with gr.Column(elem_classes=["card"]):
            gr.HTML('<div class="section-header">Analysis Status</div>')
            
            # Status indicator with the "Awaiting analysis..." text removed
            gr.HTML("""
            <div class="status-indicator" style="display: none;" id="status-indicator">
                <div class="status-dot status-waiting" id="status-light"></div>
                <div id="status-text"></div>
            </div>
            <script>
                // Update status indicator based on content
                function updateStatus() {
                    const statusBox = document.querySelector('textarea[data-testid="textbox"]');
                    const statusLight = document.getElementById('status-light');
                    const statusText = document.getElementById('status-text');
                    const statusIndicator = document.getElementById('status-indicator');
                    
                    if (statusBox && statusLight && statusText) {
                        const status = statusBox.value || '';
                        
                        if (status.includes('Processing')) {
                            statusLight.className = 'status-dot status-processing';
                            statusText.textContent = 'Processing websites...';
                            statusIndicator.style.display = 'flex';
                        } else if (status.includes('complete') || status.includes('saved')) {
                            statusLight.className = 'status-dot status-success';
                            statusText.textContent = 'Analysis completed successfully';
                            statusIndicator.style.display = 'flex';
                        } else if (status.includes('Error') || status.includes('error')) {
                            statusLight.className = 'status-dot status-error';
                            statusText.textContent = 'Error occurred during analysis';
                            statusIndicator.style.display = 'flex';
                        } else {
                            // Don't show the indicator when there's no meaningful status
                            statusIndicator.style.display = 'none';
                        }
                    }
                }
                
                // Set up observer to watch for status changes
                const setupObserver = function() {
                    const statusBox = document.querySelector('textarea[data-testid="textbox"]');
                    if (statusBox) {
                        // Initialize
                        updateStatus();
                        
                        // Create observer
                        const observer = new MutationObserver(function(mutations) {
                            updateStatus();
                        });
                        
                        // Start observing
                        observer.observe(statusBox, { 
                            attributes: true, 
                            childList: true, 
                            characterData: true,
                            subtree: true
                        });
                    } else {
                        // If element not found yet, try again in a moment
                        setTimeout(setupObserver, 500);
                    }
                };
                
                // Run on load and when DOM changes
                document.addEventListener('DOMContentLoaded', setupObserver);
                setupObserver();
            </script>
            """)
            
            status_output = gr.Textbox(label="Status", interactive=False)

        # Results output with enhanced table styling
        with gr.Column(elem_classes=["card"]):
            gr.HTML('<div class="section-header">Analysis Results</div>')
            
            # Add custom styling to make DataFrames look better
            gr.HTML("""
            <style>
                /* Apply to the DataFrame wrapper to ensure the table headers are fixed */
                [data-testid="dataframe"] .overflow-y-auto {
                    max-height: 500px !important;
                }
                
                /* Make sure the headers stay fixed */
                [data-testid="dataframe"] thead th {
                    position: sticky !important;
                    top: 0 !important;
                    z-index: 10 !important;
                    background-color: #495867 !important;
                    color: #F7F7FF !important;
                }
                
                /* Ensure the actual table takes up full width */
                [data-testid="dataframe"] table {
                    width: 100% !important;
                }
            </style>
            """)
            
            results_output = gr.DataFrame(label="Results")
            
        # Export options
        with gr.Column(elem_classes=["card"]):
            gr.HTML('<div class="section-header">Export Options</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    download_button = gr.Button("Download Results as CSV", elem_classes=["secondary"])
                    download_output = gr.File(label="Download CSV")
                
                with gr.Column(scale=1):
                    download_markdown_button = gr.Button("Download All Reports", elem_classes=["secondary"])
                    markdown_output = gr.File(label="Download Reports")
                    
                with gr.Column(scale=1):
                    download_raw_button = gr.Button("Download All Raw Content", elem_classes=["secondary"])
                    raw_content_output = gr.File(label="Download Raw Content")

        # Instructions card with fixed HTML structure
        with gr.Column(elem_classes=["instructions-card"]):
            gr.HTML('<div class="section-header">Instructions</div>')
            
            gr.HTML("""
            <ol style="list-style-type: decimal; padding-left: 1.5rem; line-height: 1.6; color: #577399; font-family: 'Aptos', sans-serif;">
                <li><strong style="color: #495867;">Enter website URLs</strong> (one per line) or upload a text file containing URLs
                   <ul style="margin-top: 0.25rem; margin-left: 1rem; list-style-type: disc;">
                      <li style="margin-top: 0.25rem;">URLs will have 'https://' and trailing slashes added automatically</li>
                   </ul>
                </li>
                <li><strong style="color: #495867;">Specify nature of business</strong> using comma-separated terms
                   <ul style="margin-top: 0.25rem; margin-left: 1rem; list-style-type: disc;">
                      <li style="margin-top: 0.25rem;">Websites must contain at least one of these terms to be accepted</li>
                   </ul>
                </li>
                <li><strong style="color: #495867;">Select nature of customers</strong> that you're looking for (B2B/B2C)
                   <ul style="margin-top: 0.25rem; margin-left: 1rem; list-style-type: disc;">
                      <li style="margin-top: 0.25rem;">Select one or more options - websites must match at least one</li>
                   </ul>
                </li>
                <li><strong style="color: #495867;">Select nature of operations</strong> that you're interested in (Manufacturing/Trading/Services)
                   <ul style="margin-top: 0.25rem; margin-left: 1rem; list-style-type: disc;">
                      <li style="margin-top: 0.25rem;">Select one or more options - websites must match at least one</li>
                   </ul>
                </li>
                <li><strong style="color: #495867;">Provide your OpenAI API key</strong> (never share your API key publicly)</li>
                <li><strong style="color: #495867;">Click "Check Websites"</strong> to process the list</li>
                <li><strong style="color: #495867;">View results</strong> in the table below - websites are only accepted if they match all your criteria</li>
                <li><strong style="color: #495867;">Export options:</strong>
                   <ul style="margin-top: 0.25rem; margin-left: 1rem; list-style-type: disc;">
                      <li style="margin-top: 0.25rem;">Download Results as CSV: Save all analysis results</li>
                      <li style="margin-top: 0.25rem;">Download All Reports: Get organized reports for each website</li>
                      <li style="margin-top: 0.25rem;">Download All Raw Content: Get the original content from all analyzed websites</li>
                   </ul>
                </li>
            </ol>
            """)
            
            gr.HTML("""
            <div style="margin-top: 15px; padding: 10px; background-color: rgba(254, 95, 85, 0.1); border-radius: 5px; border-left: 3px solid #FE5F55; font-family: 'Aptos', sans-serif;">
                <p style="margin: 0; color: #577399; font-style: italic;">Tip: A website must match ALL your criteria to be accepted. It must: 1) contain at least one business term you specified, 2) match at least one customer type, and 3) match at least one operation type.</p>
            </div>
            """)
            
        # Footer with updated colors
        gr.HTML("""
            <div class="footer">
                <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #FE5F55; margin: 0 5px;"></div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #495867; margin: 0 5px;"></div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #577399; margin: 0 5px;"></div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #BDD5EA; margin: 0 5px;"></div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #F7F7FF; margin: 0 5px; border: 1px solid #BDD5EA;"></div>
                </div>
                © 2025 Multi-Website Text Analyzer V2 | Created with Gradio
            </div>
        """)

        # Simple wrapper functions that use the processing indicator
        def process_websites_with_progress(*args):
            nonlocal all_results
            # Return success status to show progress indicator
            status_message, df, csv_file_path = process_multiple_websites(*args)
            all_results = df.to_dict('records') if df is not None else []
            # Hide progress indicator and return results
            return status_message, df, csv_file_path, gr.update(visible=False)
            
        def create_markdown_with_progress():
            report_path = create_markdown_reports()
            return report_path, gr.update(visible=False)
            
        def create_raw_content_with_progress():
            content_path = create_all_raw_content()
            return content_path, gr.update(visible=False)
            
        def show_progress_indicator():
            return gr.update(visible=True)

        # Set up event handlers with the new parameters
        submit_button.click(
            fn=show_progress_indicator,
            inputs=None,
            outputs=processing_indicator
        ).then(
            fn=process_websites_with_progress,
            inputs=[urls_input, business_terms_input, api_key_input, customer_type_input, operation_type_input],
            outputs=[status_output, results_output, csv_file, processing_indicator]
        )

        load_button.click(
            fn=upload_urls_file,
            inputs=[file_upload],
            outputs=[urls_input]
        )

        file_submit_button.click(
            fn=show_progress_indicator,
            inputs=None,
            outputs=processing_indicator
        ).then(
            fn=process_websites_with_progress,
            inputs=[urls_input, file_business_terms, file_api_key, file_customer_type, file_operation_type],
            outputs=[status_output, results_output, csv_file, processing_indicator]
        )
        
        download_button.click(
            fn=lambda f: f,
            inputs=[csv_file],
            outputs=[download_output]
        )
        
        download_markdown_button.click(
            fn=show_progress_indicator,
            inputs=None,
            outputs=processing_indicator
        ).then(
            fn=create_markdown_with_progress,
            inputs=[],
            outputs=[markdown_output, processing_indicator]
        )
        
        download_raw_button.click(
            fn=show_progress_indicator,
            inputs=None,
            outputs=processing_indicator
        ).then(
            fn=create_raw_content_with_progress,
            inputs=[],
            outputs=[raw_content_output, processing_indicator]
        )
        
    return app

# Launch the app
if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7861, share=False)
