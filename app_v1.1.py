"""
Enterprise Website Intelligence Platform with Advanced AI Workflows

Installation:
pip install langchain langchain-openai gradio requests beautifulsoup4 pandas

For enhanced workflow capabilities (optional):
pip install langgraph

If you get import errors, the system will automatically fall back to 
sequential processing with standard AI components.
"""

import os
import gradio as gr

# Set USER_AGENT to avoid warnings
os.environ.setdefault('USER_AGENT', 'LangChain-Website-Analyzer/1.0')
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import concurrent.futures
import time
import zipfile
import hashlib
import logging
import json
import tempfile
import shutil
from functools import lru_cache
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# LangChain imports - with fallbacks for compatibility
try:
    from langchain_openai import ChatOpenAI
except ImportError:
 
   from langchain.chat_models import ChatOpenAI

try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.output_parsers import JsonOutputParser, StrOutputParser
    from langchain.schema import HumanMessage, SystemMessage

try:
    from langchain.memory import ConversationBufferMemory
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
except ImportError:
    # Fallback for older versions
    ConversationBufferMemory = None
 
   InMemoryCache = None
    set_llm_cache = lambda x: None

try:
    from langchain_core.runnables.config import RunnableConfig
except ImportError:
    RunnableConfig = dict

# LangGraph imports - with fallbacks
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("LangGraph not available, using simplified workflow")
    StateGraph = None
    END = None
    MemorySaver = None
    LANGGRAPH_AVAILABLE = False

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s 
- %(message)s')
logger = logging.getLogger(__name__)

# Enable LangChain caching if available
if InMemoryCache:
    set_llm_cache(InMemoryCache())

# Data Models
class BusinessType(Enum):
    B2B = "Business to Business"
    B2C = "Business to Consumer"
    BOTH = "Both"

class OperationType(Enum):
    MANUFACTURING = "Manufacturing"
    TRADING = "Trading"
    SERVICES = "Services"

@dataclass
class WebsiteData:
    url: str
    text: str
    soup: Optional[BeautifulSoup] = None
    html_content: Optional[str] = None
    meta_info: Dict[str, str] = None
    
@dataclass
class BusinessAnalysis:
    business_nature: 
str
    customer_type: str
    operation_type: str
    countries: str
    products_services: str
    target_customers: str
    target_industries: str
    business_model: str
    global_presence: str

@dataclass
class AnalysisResult:
    website: str
    status: str
    company_description: str
    business_analysis: BusinessAnalysis
    decision: str
    keyword_matches: List[str]
    processing_time: float
    error_message: str = ""

# LangGraph State
class AnalysisState:
    def __init__(self):
        self.website_data: 
Optional[WebsiteData] = None
        self.translated_text: str = ""
        self.company_description: str = ""
        self.business_analysis: Optional[BusinessAnalysis] = None
        self.keyword_matches: List[str] = []
        self.decision: str = ""
        self.error_message: str = ""
        self.processing_time: float = 0.0

# LangChain Components
class AdvancedWebsiteAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
       
 self.api_key = api_key
        self.model = model
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=1500,
            request_timeout=30
        )
        
 
       # Initialize memory and cache
        self.memory = ConversationBufferMemory(return_messages=True) if ConversationBufferMemory else None
        self.cache = {}
        self.session_locks = {}
        
        # Initialize prompts
        self._setup_prompts()
        
        # Initialize chains
        self._setup_chains()
       
 
        # Initialize graph
        self._setup_workflow()
    
    def _setup_prompts(self):
        """Setup LangChain prompt templates"""
        
        # Translation prompt
        self.translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional translator.
Translate text to English while preserving formatting and business context."),
            ("human", "Translate the following text to English:\n\n{text}")
        ])
        
        # Company description prompt
        self.description_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business analyst. Based on website content, provide a concise 15-word description of what this company does. 
    
        If insufficient information, state 'Insufficient information available to determine company purpose.'"""),
            ("human", "Website: {url}\n\nContent: {content}\n\nProvide a concise description:")
        ])
        
        # Business analysis prompt
        self.business_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business intelligence analyst. Analyze the website content and provide structured business 
information.
            
            Respond with ONLY a JSON object containing these exact fields:
            {{
                "business_nature": "Detailed description of what the company does",
                "customer_type": "Business to Business" OR "Business to Consumer" OR "Both",
       
         "operation_type": "Manufacturing" OR "Trading" OR "Services" OR combinations like "Manufacturing, Trading",
                "countries": "List of countries/regions mentioned or 'Not available'",
                "products_services": "Detailed list of products/services offered",
                "target_customers": "Description of their target customer base",
               
 "target_industries": "Industries they serve",
                "business_model": "How they generate revenue",
                "global_presence": "International operations and presence"
            }}
            
            Base your analysis only on the provided content.
Be specific and detailed."""),
            ("human", "Website: {url}\n\nContent: {content}\n\nAnalyze this business:")
        ])
        
        # Detailed analysis prompt for additional pages
        self.detailed_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business intelligence expert. Analyze multiple pages from a company website and provide comprehensive insights.
           
 
            Focus on extracting:
            1. Specific products and services
            2. Target market segments
            3. Business model details
            4. Global operations
            5. Industry expertise
           
 
            Provide detailed, factual information based on the content."""),
            ("human", """Main Website: {main_url}
            
            Content from multiple pages:
            {combined_content}
            
            Provide detailed 
analysis for:
            1. Products & Services:
            2. Target Customers:
            3. Target Industries:
            4. Business Model:
            5. Global Presence:""")
        ])
    
    def _setup_chains(self):
        """Setup LangChain processing 
chains"""
        
        # Translation chain
        self.translation_chain = (
            self.translation_prompt 
            |
self.llm 
            |
StrOutputParser()
        )
        
        # Description chain
        self.description_chain = (
            self.description_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Business 
analysis chain with error handling
        try:
            self.business_analysis_chain = (
                self.business_analysis_prompt
                | self.llm
                | JsonOutputParser()
            )
        except Exception as e:
 
           logger.warning(f"JSON parser setup failed: {e}, using string parser")
            self.business_analysis_chain = (
                self.business_analysis_prompt
                | self.llm
                | StrOutputParser()
            )
      
  
        # Detailed analysis chain
        self.detailed_analysis_chain = (
            self.detailed_analysis_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_workflow(self):
        """Setup LangGraph workflow with fallback"""
        
  
      if not LANGGRAPH_AVAILABLE:
            logger.info("Using simplified sequential workflow (LangGraph not available)")
            self.graph = None
            return
        
        # Define the graph
        workflow = StateGraph(dict) # Using dict as a placeholder for actual State class if not defined inline
        
        # Add nodes
 
       workflow.add_node("extract_content", self._extract_content_node)
        workflow.add_node("translate_text", self._translate_node)
        workflow.add_node("generate_description", self._description_node)
        workflow.add_node("analyze_business", self._business_analysis_node)
        workflow.add_node("detailed_analysis", self._detailed_analysis_node)
        workflow.add_node("make_decision", self._decision_node)
        workflow.add_node("finalize_result", self._finalize_node)
        
        # Define edges
        workflow.set_entry_point("extract_content")
        workflow.add_edge("extract_content", "translate_text")
   
     workflow.add_edge("translate_text", "generate_description")
        workflow.add_edge("generate_description", "analyze_business")
        workflow.add_edge("analyze_business", "detailed_analysis")
        workflow.add_edge("detailed_analysis", "make_decision")
        workflow.add_edge("make_decision", "finalize_result")
        workflow.add_edge("finalize_result", END)
        
        # Compile the graph
        try:
            self.graph = workflow.compile(checkpointer=MemorySaver()) if MemorySaver else workflow.compile()
        
except Exception as e:
            logger.warning(f"Failed to compile LangGraph workflow: {e}, using fallback")
            self.graph = None
    
    def _extract_content_node(self, state: dict) -> dict:
        """Extract content from website"""
        try:
            url = state["url"]
            start_time = time.time()
    
        
            # Use existing extraction logic
            extracted_text, soup, html_content = self._extract_text_from_url(url)
            
            if extracted_text is not None and extracted_text.startswith("Error"):
                state["error_message"] = extracted_text
                state["status"] 
= "Error"
                return state
            
            # Extract meta information
            meta_info = self._extract_meta_info(soup) if soup else {}
            
            state["website_data"] = {
           
     "url": url,
                "text": extracted_text,
                "soup": soup,
                "html_content": html_content,
                "meta_info": meta_info
            }
            state["processing_time"] = 
time.time() - start_time
            
        except Exception as e:
            logger.error(f"Content extraction error: {str(e)}")
            state["error_message"] = f"Content extraction error: {str(e)}"
            state["status"] = "Error"
        
        return state
    
    def _translate_node(self, state: dict) 
-> dict:
        """Translate content if needed"""
        try:
            website_data = state.get("website_data")
            if not website_data or state.get("error_message"):
                return state
            
            text = website_data["text"]
       
     
            # Check if translation is needed
            if self._is_likely_english(text):
                state["translated_text"] = text
                state["translation_status"] = "No translation needed"
            else:
               
 # Use LangChain for translation
                translated = self.translation_chain.invoke({"text": text[:2000]}) # Limiting text to avoid excessive API usage
                state["translated_text"] = translated
                state["translation_status"] = "Translation successful"
                
        except Exception as e:
            
logger.error(f"Translation error: {str(e)}")
            state["translated_text"] = website_data["text"] if website_data else ""
            state["translation_status"] = f"Translation error: {str(e)}"
        
        return state
    
    def _description_node(self, state: dict) -> dict:
        """Generate company description"""
        try:
            if state.get("error_message"):
   
             return state
            
            website_data = state.get("website_data")
            translated_text = state.get("translated_text", "")
            
            if not translated_text:
                state["company_description"] = "Unable 
to generate description"
                return state
            
            # Use LangChain for description generation
            description = self.description_chain.invoke({
                "url": website_data["url"],
                "content": translated_text[:1000] # Limiting content for brevity
    
        })
            
            state["company_description"] = description.strip()
            
        except Exception as e:
            logger.error(f"Description generation error: {str(e)}")
            state["company_description"] = f"Error generating description: {str(e)}"
        
   
     return state
    
    def _business_analysis_node(self, state: dict) -> dict:
        """Analyze business using LangChain"""
        try:
            if state.get("error_message"):
                return state
            
            website_data = state.get("website_data")
      
      translated_text = state.get("translated_text", "")
            
            if not translated_text:
                state["business_analysis"] = self._get_default_analysis()
                return state
            
            # Use LangChain for business analysis
 
           analysis_result = self.business_analysis_chain.invoke({
                "url": website_data["url"],
                "content": translated_text[:2500] # Limiting content to manage token usage
            })
            
            # Handle both JSON and string responses
           
 if isinstance(analysis_result, dict):
                # Direct JSON response
                business_analysis = BusinessAnalysis(
                    business_nature=analysis_result.get("business_nature", "Unable to determine"),
                    customer_type=analysis_result.get("customer_type", "Unable to determine"),
              
      operation_type=analysis_result.get("operation_type", "Unable to determine"),
                    countries=analysis_result.get("countries", "Not available"),
                    products_services=analysis_result.get("products_services", "Not available"),
                    target_customers=analysis_result.get("target_customers", "Not available"),
                    target_industries=analysis_result.get("target_industries", "Not available"),
   
                 business_model=analysis_result.get("business_model", "Not available"),
                    global_presence=analysis_result.get("global_presence", "Not available")
                )
            else:
                # String response - parse manually
          
      business_analysis = self._parse_string_analysis(str(analysis_result))
            
            state["business_analysis"] = business_analysis
            
        except Exception as e:
            logger.error(f"Business analysis error: {str(e)}")
            state["business_analysis"] = self._get_default_analysis()
        
      
  return state
    
    def _parse_string_analysis(self, analysis_text: str) -> BusinessAnalysis:
        """Parse string analysis response to BusinessAnalysis object"""
        try:
            # Try to extract JSON from string
            import json
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
  
              analysis_dict = json.loads(json_match.group())
                return BusinessAnalysis(
                    business_nature=analysis_dict.get("business_nature", "Unable to determine"),
                    customer_type=analysis_dict.get("customer_type", "Unable to determine"),
                    operation_type=analysis_dict.get("operation_type", 
"Unable to determine"),
                    countries=analysis_dict.get("countries", "Not available"),
                    products_services=analysis_dict.get("products_services", "Not available"),
                    target_customers=analysis_dict.get("target_customers", "Not available"),
                    target_industries=analysis_dict.get("target_industries", "Not available"),
          
          business_model=analysis_dict.get("business_model", "Not available"),
                    global_presence=analysis_dict.get("global_presence", "Not available")
                )
        except Exception as e:
            logger.warning(f"Failed to parse JSON from string: {e}")
        
        # Fallback: extract from text 
patterns
        return BusinessAnalysis(
            business_nature=self._extract_field_from_text(analysis_text, "business_nature") or "Unable to determine",
            customer_type=self._extract_field_from_text(analysis_text, "customer_type") or "Unable to determine",
            operation_type=self._extract_field_from_text(analysis_text, "operation_type") or "Unable to determine",
            countries=self._extract_field_from_text(analysis_text, "countries") or "Not available",
            products_services=self._extract_field_from_text(analysis_text, "products_services") or "Not available",
        
    target_customers=self._extract_field_from_text(analysis_text, "target_customers") or "Not available",
            target_industries=self._extract_field_from_text(analysis_text, "target_industries") or "Not available",
            business_model=self._extract_field_from_text(analysis_text, "business_model") or "Not available",
            global_presence=self._extract_field_from_text(analysis_text, "global_presence") or "Not available"
        )
    
    def _extract_field_from_text(self, text: str, field_name: str) -> Optional[str]:
        """Extract field value from text using patterns"""
       
 patterns = [
            rf'"{field_name}":\s*"([^"]*)"',  # JSON-like "field": "value"
            rf'{field_name}:\s*"([^"]*)"',    # Field: "Value"
            rf'{field_name}:\s*([^\n,}}]*)', # Field: Value (until newline, comma, or end of a structure)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
     
           return match.group(1).strip()
        
        return None
    
    def _detailed_analysis_node(self, state: dict) -> dict:
        """Perform detailed analysis of additional pages"""
        try:
            if state.get("error_message"):
                return state
       
     
            website_data = state.get("website_data")
            translated_text = state.get("translated_text", "")
            
            # Extract additional pages
            soup = website_data.get("soup")
            if soup:
            
    about_links, product_links = self._extract_key_page_links(soup, website_data["url"])
                
                additional_content = ""
                # Limiting to 2 additional pages to manage processing time and complexity
                for link, text in (about_links + product_links)[:2]:  
                    try:
         
               page_text, _, _ = self._extract_text_from_url(link)
                        if page_text is not None and not page_text.startswith("Error"):
                            additional_content += f"\n\nPage: {link}\nContent: {page_text[:1000]}" # Limiting content per page
                    except Exception as 
e:
                        logger.warning(f"Error fetching additional page {link}: {str(e)}")
                
                if additional_content:
                    # Use detailed analysis chain
              
      detailed_result = self.detailed_analysis_chain.invoke({
                        "main_url": website_data["url"],
                        "combined_content": translated_text[:1500] + additional_content # Limiting combined content
                    })
                    

                    # Update business analysis with detailed information
                    business_analysis = state.get("business_analysis")
                    if business_analysis and detailed_result:
                        # Parse detailed result and 
update fields
                        self._update_analysis_with_details(business_analysis, detailed_result)
                        state["business_analysis"] = business_analysis
            
        except Exception as e:
            logger.error(f"Detailed analysis error: {str(e)}")
        
  
      return state
    
    def _decision_node(self, state: dict) -> dict:
        """Make final decision based on criteria"""
        try:
            if state.get("error_message"):
                state["decision"] = "Error"
                return state
           
 
            business_analysis = state.get("business_analysis")
            translated_text = state.get("translated_text", "")
            business_terms = state.get("business_terms", [])
            customer_types = state.get("customer_types", [])
            operation_types = state.get("operation_types", [])

            if not business_analysis: # Ensure business_analysis exists
                state["decision"] = "Error - Missing Business Analysis"
                return state
            
            # 
Check keyword matches
            keyword_matches = []
            text_lower = translated_text.lower()
            for term in business_terms:
                if term.lower() in text_lower:
                    keyword_matches.append(term)
            
    
        state["keyword_matches"] = keyword_matches
            
            # Apply decision logic
            customer_match = self._check_customer_type_match(business_analysis.customer_type, customer_types)
            operation_match = self._check_operation_type_match(business_analysis.operation_type, operation_types)
            terms_match = len(keyword_matches) > 0 if business_terms else True
            
 
           if customer_match and operation_match and terms_match:
                state["decision"] = "Accepted"
            else:
                rejection_reasons = []
                if not customer_match:
                  
  rejection_reasons.append("Customer Type Mismatch")
                if not operation_match:
                    rejection_reasons.append("Operation Type Mismatch")
                if not terms_match and business_terms: # Only a reason if terms were specified
                    rejection_reasons.append("No Business Terms Found")
               
 
                state["decision"] = f"Rejected ({', '.join(rejection_reasons)})" if rejection_reasons else "Rejected"
            
        except Exception as e:
            logger.error(f"Decision error: {str(e)}")
            state["decision"] = "Error in decision making"
        
        return state
    
    
def _finalize_node(self, state: dict) -> dict:
        """Finalize the analysis result"""
        try:
            website_data = state.get("website_data", {})
            business_analysis = state.get("business_analysis")
            
            # Create final result
            result = AnalysisResult(
      
          website=website_data.get("url", "Unknown"),
                status="Success" if not state.get("error_message") else "Error",
                company_description=state.get("company_description", "N/A"),
                business_analysis=business_analysis or self._get_default_analysis(),
                decision=state.get("decision", "Unknown"),
                
keyword_matches=state.get("keyword_matches", []),
                processing_time=state.get("processing_time", 0.0),
                error_message=state.get("error_message", "")
            )
            
            state["final_result"] = result
            
        except Exception as e:
    
        logger.error(f"Finalization error: {str(e)}")
            # Ensure error_message is set in the state if finalization fails
            state["error_message"] = state.get("error_message", "") + f" | Finalization error: {str(e)}"
            # Create a default error result if finalization fails badly
            state["final_result"] = AnalysisResult(
                website=website_data.get("url", "Unknown"),
                status="Error",
                company_description="N/A",
                business_analysis=self._get_default_analysis(),
                decision="Error",
                keyword_matches=[],
                processing_time=state.get("processing_time", 0.0),
                error_message=state.get("error_message", "")
            )
        
        return state
    
    def analyze_website(self, url: str, business_terms: List[str] = None, 
                       customer_types: List[str] = None, 
              
         operation_types: List[str] = None) -> AnalysisResult:
        """Analyze a single website using LangGraph workflow or fallback"""
        
        # Prepare initial state
        initial_state = {
            "url": url,
            "business_terms": business_terms or [],
            "customer_types": 
customer_types or [],
            "operation_types": operation_types or [],
            "error_message": "" # Initialize error message
        }
        
        try:
            if self.graph and LANGGRAPH_AVAILABLE:
                # Use LangGraph workflow
                config = {"configurable": {"thread_id": f"analysis_{hashlib.md5(url.encode()).hexdigest()}"}}
     
           final_state = self.graph.invoke(initial_state, config)
                return final_state.get("final_result")
            else:
                # Use sequential fallback processing
                return self._sequential_analysis(initial_state)
                
    
    except Exception as e:
            logger.error(f"Analysis error for {url}: {str(e)}")
            return AnalysisResult(
                website=url,
                status="Error",
                company_description="Analysis execution error",
              
  business_analysis=self._get_default_analysis(),
                decision="Error",
                keyword_matches=[],
                processing_time=0.0,
                error_message=str(e)
            )
    
    def _sequential_analysis(self, initial_state: dict) -> AnalysisResult:
        """Fallback 
sequential analysis when LangGraph is not available"""
        start_time = time.time() # Define start_time at the beginning of the try block
        try:
            # Sequential processing
            state = initial_state.copy()
            state = self._extract_content_node(state)
            if state.get("error_message"):
     
           return self._create_error_result(state, time.time() - start_time)
            
            state = self._translate_node(state)
            state = self._description_node(state)
            state = self._business_analysis_node(state)
            state = self._detailed_analysis_node(state)
            state = self._decision_node(state)
   
         state = self._finalize_node(state)
            
            final_result = state.get("final_result")
            if final_result: # Ensure processing_time is updated if not set in _finalize_node error path
                 if final_result.processing_time == 0.0 and "processing_time" not in initial_state: # Check if it was the initial calculation
                    final_result.processing_time = time.time() - start_time
            return final_result
            
        except Exception as e:
            logger.error(f"Sequential analysis error: {str(e)}")
            return AnalysisResult(
             
   website=initial_state.get("url", "Unknown"),
                status="Error",
                company_description="Sequential analysis error",
                business_analysis=self._get_default_analysis(),
                decision="Error",
                keyword_matches=[],
              
  processing_time=time.time() - start_time, # Calculate processing time here for the exception case
                error_message=str(e)
            )
    
    def _create_error_result(self, state: dict, processing_time: float) -> AnalysisResult:
        """Create error result from state"""
        return AnalysisResult(
            website=state.get("url", "Unknown"),
            status="Error",
       
     company_description="Analysis error", # More generic description for early errors
            business_analysis=self._get_default_analysis(),
            decision="Error",
            keyword_matches=[],
            processing_time=processing_time,
            error_message=state.get("error_message", "Unknown error")
        )
    
    # Helper methods (keeping original functionality)
    def _extract_text_from_url(self, url: str) -> Tuple[Optional[str], Optional[BeautifulSoup], 
Optional[str]]:
        """Extract text from URL (keeping original logic)"""
        try:
            formatted_url = self._format_url(url)
            headers = {
                'User-Agent': os.environ.get('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'), # Use configured USER_AGENT
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', # More comprehensive Accept header
                'Accept-Language': 'en-US,en;q=0.5', # Simpler Accept-Language
                'Connection': 'keep-alive',
                'Cache-Control': 'max-age=0' # Request fresh content
            }
            
      
      response = requests.get(formatted_url, headers=headers, timeout=15, verify=True) # Increased timeout
            response.raise_for_status() # Check for HTTP errors
            
            # Attempt to detect encoding, fallback to utf-8
            content_type = response.headers.get('content-type', '').lower()
            if 'charset=' in content_type:
                encoding = response.encoding
            else:
                encoding = 'utf-8' # Default if not specified

            soup = BeautifulSoup(response.content, 'html.parser', from_encoding=encoding) # Use response.content and specify encoding

            for element in soup(["script", "style", "nav", "footer", "iframe", "header", "aside", "form"]): # Added more common noise tags
                element.decompose() # Use decompose for cleaner removal
            
  
          text = soup.get_text(separator='\n', strip=True) # strip=True for cleaner lines
            # Further clean up: remove multiple newlines and leading/trailing whitespace from overall text
            text = re.sub(r'\n\s*\n', '\n', text).strip()

            return text, soup, response.text # response.text might differ if encoding was guessed by requests
            
        except requests.exceptions.RequestException as e: # Specific exception handling
            logger.error(f"Request error extracting text from {url}: {str(e)}")
            return f"Error: Network or HTTP error ({type(e).__name__})", None, None
        except Exception as 
e:
            logger.error(f"Error extracting text from {url}: {str(e)}")
            return f"Error: {str(e)}", None, None
    
    def _format_url(self, url: str) -> str:
        """Format URL properly"""
        url = url.strip()
        if not re.match(r'^(?:http|ftp)s?://', url): # More robust check for scheme
            url = 'https://' + url
       
 # Removed trailing slash addition as it's not always desired and can sometimes break URLs
        # if not url.endswith('/'):
        #     url = url + '/'
        return url
    
    def _is_likely_english(self, text: str, sample_size: int = 500) -> bool:
        """Check if text is likely in English"""
        if not text or text.isspace(): # Handle empty or whitespace-only text
            return True # Assume English or no content to check

        # More comprehensive set of common English words
        common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 
                   
        'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 
                           'from', 'they', 'we', 'say', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what'])
        sample = text[:sample_size].lower()
        words = re.findall(r'\b\w+\b', sample) # \w+ matches alphanumeric characters
        if not words:
            return True # No words found, could be non-alphabetic script or empty after cleaning
       
 english_count = sum(1 for word in words if word in common_words)
        # Adjusted threshold and consider very short texts
        if len(words) < 5: # For very short texts, higher proportion needed
            return (english_count / len(words) > 0.4)
        return (english_count / len(words) > 0.15) # Original threshold for longer samples
    
    def _extract_meta_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract meta information from soup"""
        meta_info = {}
        if not soup:
            return meta_info
            
 
       if soup.title and soup.title.string: # Ensure title has a string
            meta_info['title'] = soup.title.string.strip()
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', '')).lower() # Also check 'property' for OpenGraph tags
            content = meta.get('content', '')
            if name in ['description', 'keywords', 'og:title', 'og:description'] and content: # Added OpenGraph meta tags
           
     meta_info[name] = content.strip()
                
        return meta_info
    
    def _extract_key_page_links(self, soup: BeautifulSoup, base_url: str, max_links: int = 3) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Extract about and product page links"""
        if not soup:
            return [], []
         
   
        about_links_scored = []
        product_links_scored = []
        
        # More comprehensive and weighted terms
        about_terms = {'about': 3, 'company': 2, 'who we are': 3, 'our story': 2, 'profile': 2, 'mission': 2, 'vision': 2, 'team': 2}
        product_terms = {'product': 3, 'service': 3, 'solution': 3, 'offering': 2, 'what we do': 2, 'portfolio': 2, 'platform': 2, 'cases': 2, 'technologies': 2}
        
        processed_hrefs = set() # To avoid duplicate links

        for link in soup.find_all('a', href=True):
           
 href = link.get('href', '')
            link_text = link.get_text().lower().strip()
            
            if not href or href.startswith(('#', 'javascript:', 'mailto:')): # Skip anchors, JS calls, mailto
                continue
                
            # Create absolute URL
            try:
                abs_href = requests.compat.urljoin(base_url, href)
            except ValueError:
                continue # Skip malformed URLs

            # Normalize URL to avoid processing slight variations of the same page
            normalized_href = abs_href.split('#')[0].rstrip('/') 
            if normalized_href in processed_hrefs or not normalized_href.startswith(base_url.split('/')[0] + '//' + base_url.split('/')[2]): # Basic check if it's an internal link
                continue
            processed_hrefs.add(normalized_href)

            # Skip common social media, utility links
            skip_patterns = ['twitter.com', 'facebook.com', 
'linkedin.com', 'instagram.com', 'youtube.com',
                             'login', 'signin', 'contact', 'privacy', 'terms', 'support', 'faq', 'blog', 'news', 'career', 'download', 'press']
            if any(skip in normalized_href.lower() for skip in skip_patterns) or any(skip in link_text for skip in ['contact', 'support', 'login']):
                continue
            
            # Scoring based on terms in href and link text
            current_link_text = link_text[:100] # Limit link text for scoring
            href_lower = normalized_href.lower()

            about_score = sum(weight for term, weight in about_terms.items() if term in href_lower or term in current_link_text)
            product_score = sum(weight for term, weight in product_terms.items() if term in href_lower or term in current_link_text)
            
            # Prioritize links with descriptive text
            if len(current_link_text) > 5:
                about_score +=1
                product_score +=1

            if about_score > 0:
                about_links_scored.append((normalized_href, link_text, about_score))
            elif product_score > 0:
   
             product_links_scored.append((normalized_href, link_text, product_score))
        
        # Sort by score and return unique links
        about_links_scored.sort(key=lambda x: x[2], reverse=True)
        product_links_scored.sort(key=lambda x: x[2], reverse=True)
        
        # Deduplicate again after sorting, preferring higher scores
        final_about_links = []
        seen_urls = set()
        for link_tuple in about_links_scored:
            if link_tuple[0] not in seen_urls and len(final_about_links) < max_links :
                final_about_links.append((link_tuple[0], link_tuple[1]))
                seen_urls.add(link_tuple[0])

        final_product_links = []
        for link_tuple in product_links_scored:
            if link_tuple[0] not in seen_urls and len(final_product_links) < max_links:
                final_product_links.append((link_tuple[0], link_tuple[1]))
                seen_urls.add(link_tuple[0])
        
        return final_about_links, final_product_links
    
    def _get_default_analysis(self) -> BusinessAnalysis:
        """Get default business analysis"""
 
       return BusinessAnalysis(
            business_nature="Unable to determine",
            customer_type="Unable to determine",
            operation_type="Unable to determine",
            countries="Not available",
            products_services="Not available",
            target_customers="Not available",
           
 target_industries="Not available",
            business_model="Not available",
            global_presence="Not available"
        )
    
    def _update_analysis_with_details(self, analysis: BusinessAnalysis, detailed_result: str):
        """Update analysis with detailed information from combined page content."""
        # This method attempts to parse a structured string and update the BusinessAnalysis object.
        # It looks for specific headings in the `detailed_result` string.
        
        def extract_section(text: str, heading: str) -> str:
            """Helper to extract content under a specific heading."""
            try:
                # Regex to find heading and capture content until the next heading or end of string
                # Assumes headings are followed by a colon and on their own line, or start of line.
                match = re.search(rf"^\s*{re.escape(heading)}:?\s*\n?(.*?)(?=\n\s*\w+\s*:|\Z)", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                if match:
                    return match.group(1).strip()
            except Exception: # pylint: disable=broad-except
                # In case of regex error or other issues, return empty.
                pass
            return ""

        # Update products/services if new information is more detailed
        products_text = extract_section(detailed_result, "Products & Services")
     # Check if "Products & Services:" in detailed_result: # Original check
        if products_text and (analysis.products_services == "Not available" or len(products_text) > len(analysis.products_services)):
            analysis.products_services = products_text

        # Update target customers
        customers_text = extract_section(detailed_result, "Target Customers")
        if customers_text and (analysis.target_customers == "Not available" or len(customers_text) > len(analysis.target_customers)):
            analysis.target_customers = customers_text

        # Update target industries
        industries_text = extract_section(detailed_result, "Target Industries")
        if industries_text and (analysis.target_industries == "Not available" or len(industries_text) > len(analysis.target_industries)):
            analysis.target_industries = industries_text

        # Update business model
        model_text = extract_section(detailed_result, "Business Model")
        if model_text and (analysis.business_model == "Not available" or len(model_text) > len(analysis.business_model)):
            analysis.business_model = model_text

        # Update global presence
        presence_text = extract_section(detailed_result, "Global Presence")
        if presence_text and (analysis.global_presence == "Not available" or len(presence_text) > len(analysis.global_presence)):
            analysis.global_presence = presence_text

    
    def _check_customer_type_match(self, business_customer_type: str, filter_customer_types: List[str]) -> bool:
        """Check if customer type matches filter"""
        if not filter_customer_types: # If no filter is provided, it's a match
    
        return True
        
        if business_customer_type == "Unable to determine": # If undetermined, it cannot match a specific filter
            return False
        
        # If the business is 'Both', it matches if 'Both' is in the filter, or if its constituent types are.
        if business_customer_type == BusinessType.BOTH.value: # Use Enum value for comparison
            return BusinessType.BOTH.value in filter_customer_types or \
                   BusinessType.B2B.value in filter_customer_types or \
                   BusinessType.B2C.value in filter_customer_types
        
     
   return business_customer_type in filter_customer_types
    
    def _check_operation_type_match(self, business_operation_type: str, filter_operation_types: List[str]) -> bool:
        """Check if operation type matches filter"""
        if not filter_operation_types: # If no filter is provided, it's a match
            return True
        
        if business_operation_type == "Unable to determine": # If undetermined, it cannot match
            return False
        
        # Business operation type can be a comma-separated list (e.g., "Manufacturing, Services")
       business_ops_list = {op.strip() for op in business_operation_type.split(',')}
        
        # Check if any of the business's operation types are in the filter list
        return any(filtered_op in business_ops_list for filtered_op in filter_operation_types)


# Enhanced Streaming Processor - FIXED VERSION
class StreamingWebsiteProcessor:
    def __init__(self, api_key: str, max_workers: int = 4):
        self.analyzer = AdvancedWebsiteAnalyzer(api_key)
        self.max_workers = max_workers
        self.results: List[AnalysisResult] = [] # Ensure results is typed
        
    def process_websites_stream(self, urls: List[str], business_terms: Optional[List[str]] = None, # Optional with default None
 
                              customer_types: Optional[List[str]] = None, 
                               operation_types: Optional[List[str]] = None):
        """Process websites with streaming results using ThreadPoolExecutor - SIMPLIFIED VERSION"""
        
       
 # FIX: Clear previous results to avoid accumulation across runs
        self.results = [] # Clears results at the start of a new processing batch
        
        total_urls = len(urls)
        if total_urls == 0: # Handle empty URL list
            yield "ℹ️ No URLs to process.", None, None, "No URLs provided for analysis.", True
            return

        # Ensure default empty lists if None is passed
        effective_business_terms = business_terms or []
        effective_customer_types = customer_types or []
        effective_operation_types = operation_types or []
        
        def analyze_single_website(url: str) -> AnalysisResult:
            # Passes the effective (defaulted if None) lists to the analyzer
            return self.analyzer.analyze_website(url, effective_business_terms, effective_customer_types, effective_operation_types)
        
        # Process websites concurrently
       
 with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(analyze_single_website, url): url for url in urls}
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
  
              completed_count += 1
                
                try:
                    result = future.result()
                    self.results.append(result)
          
          
                    # Yield progress update
                    progress = int((completed_count / total_urls) * 100)
                    # UX: Clear and informative status update during processing
                    status_message = f"🔄 PROCESSING ({completed_count}/{total_urls}) - {progress}% Complete"
              
      
                    # FIX: Don't yield DataFrames here, just status and log. DataFrame will be built from self.results
                    yield status_message, None, None, f"Completed: {url}", False # is_final is False here
                    
                except Exception as 
e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    error_result = AnalysisResult( # Consistent error result structure
                        website=url,
                        status="Error",
       
                 company_description="Processing error encountered", # UX: More user-friendly error message
                        business_analysis=self.analyzer._get_default_analysis(), # Fallback to default analysis
                        decision="Error",
                        keyword_matches=[],
          
              processing_time=0.0, # Processing time might not be available
                        error_message=str(e)
                    )
                    self.results.append(error_result)
                    # UX: Clear error status for the specific URL
  
                  yield f"❌ ERROR ({completed_count}/{total_urls}) - {url}", None, None, f"Error processing: {url} - {str(e)}", False
        
        # Final completion signal
        accepted_count = sum(1 for r in self.results if r.decision == "Accepted")
        rejected_count = sum(1 for r in self.results if r.decision.startswith("Rejected"))
        errors_count = sum(1 for r in self.results if r.status == "Error")
   
     
        # UX: Comprehensive final status message with emojis for quick visual assessment
        final_status_message = f"🎉 Analysis Complete! {accepted_count} Accepted, {rejected_count} Rejected, {errors_count} Errors"

        
        # Generate CSV file path (but content comes from self.results later)
        csv_file_path = self._save_results_to_csv() # This now uses self.results
        
        # FIX: Signal completion with final status and path to the generated CSV
        yield final_status_message, None, csv_file_path, "Analysis completed. You can now download reports.", True # is_final is True
    
    def _result_to_dataframe_row(self, result: AnalysisResult) -> pd.DataFrame:
        """Convert analysis result to DataFrame row"""
       # UX: Ensures consistent data representation for each row in the output table.
       # Replaces newlines to prevent breaking CSV or table display.
 row_data = {
            "Website": result.website,
            "Decision": result.decision,
            "Description": result.company_description.replace('\n', ' ').replace('\r', ''),
            "Business": result.business_analysis.business_nature.replace('\n', ' ').replace('\r', ''),
            "Customers": result.business_analysis.customer_type,
            "Operations": result.business_analysis.operation_type,
            "Countries": 
result.business_analysis.countries.replace('\n', ' ').replace('\r', ''),
            "Products": result.business_analysis.products_services.replace('\n', ' ').replace('\r', ''),
            "Industries": result.business_analysis.target_industries.replace('\n', ' ').replace('\r', ''),
            "Model": result.business_analysis.business_model.replace('\n', ' ').replace('\r', ''),
            "Global": result.business_analysis.global_presence.replace('\n', ' ').replace('\r', ''),
            "Keywords": ", ".join(result.keyword_matches) if result.keyword_matches else "None",
            "Status": result.status,
  
          "Time": f"{result.processing_time:.1f}s" # UX: Formatted processing time for readability
        }
        return pd.DataFrame([row_data])
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame"""
        if not self.results:
            return pd.DataFrame() # UX: Handles empty results gracefully by returning an empty DataFrame.
        
        rows = []
        # UX: Iterates through all processed results to build a comprehensive DataFrame.
for result in self.results:
            row_data = {
                "Website": result.website,
                "Decision": result.decision,
                "Description": result.company_description.replace('\n', ' ').replace('\r', ''),
                "Business": result.business_analysis.business_nature.replace('\n', ' ').replace('\r', ''),
         
       "Customers": result.business_analysis.customer_type,
                "Operations": result.business_analysis.operation_type,
                "Countries": result.business_analysis.countries.replace('\n', ' ').replace('\r', ''),
                "Products": result.business_analysis.products_services.replace('\n', ' ').replace('\r', ''),
                "Industries": result.business_analysis.target_industries.replace('\n', ' ').replace('\r', ''),
               
 "Model": result.business_analysis.business_model.replace('\n', ' ').replace('\r', ''),
                "Global": result.business_analysis.global_presence.replace('\n', ' ').replace('\r', ''),
                "Keywords": ", ".join(result.keyword_matches) if result.keyword_matches else "None",
                "Status": result.status,
                "Time": f"{result.processing_time:.1f}s"
            }
       
     rows.append(row_data)
        
        return pd.DataFrame(rows)
    
    def _save_results_to_csv(self) -> Optional[str]: # Return type hint
        """Save results to CSV file"""
        if not self.results: # Do not create an empty CSV if no results
            return None
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # UX: Creates a temporary directory for clean file management.
            temp_dir = tempfile.mkdtemp(prefix="enterprise_analysis_")
            csv_filename = os.path.join(temp_dir, f"enterprise_analysis_results_{timestamp}.csv")
    
        
            df = self._results_to_dataframe()
            if df.empty: # Double check if dataframe is empty
                 return None
            # UX: Standard CSV saving with proper escaping for data integrity.
            df.to_csv(csv_filename, index=False, escapechar='\\', quoting=1) # Using QUOTE_ALL from csv module via pandas
            
            return csv_filename if os.path.exists(csv_filename) else None
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
        
    return None
    
    def _save_detailed_reports(self) -> Optional[str]: # Return type hint
        """Generate and save detailed reports as ZIP file"""
        if not self.results: # Do not create an empty ZIP if no results
            return None
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            temp_dir = tempfile.mkdtemp(prefix="detailed_reports_")
            zip_filename = os.path.join(temp_dir, f"detailed_reports_{timestamp}.zip")
            
            # UX: Compresses multiple reports into a single ZIP file for easy download.
      with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Generate individual reports for each website
                for i, result in enumerate(self.results):
                    report_content = self._generate_individual_report(result)
                    # UX: Sanitizes website URL for use as a filename.
                    safe_website_name = re.sub(r'[<>:"/\\|?*]', '_', result.website)
                    report_name = f"report_{i+1}_{safe_website_name[:50]}.txt" # Truncate long names
 
                   zipf.writestr(report_name, report_content)
                
                # Generate summary report
                summary_content = self._generate_summary_report()
                zipf.writestr("summary_report.txt", summary_content)
          
      
                # Add CSV to the zip
                df = self._results_to_dataframe()
                if not df.empty:
                    csv_content = df.to_csv(index=False, escapechar='\\', quoting=1)
                    zipf.writestr("analysis_results.csv", csv_content)
            
        
    return zip_filename if os.path.exists(zip_filename) else None
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            return None
    
    def _generate_individual_report(self, result: AnalysisResult) -> str:
        """Generate detailed report for individual website"""
        # UX: Structures the individual report with clear headings for readability.
        # Provides a comprehensive overview of the analysis for a single website.
        report = f"""
WEBSITE ANALYSIS REPORT
======================

Website: {result.website}
Status: {result.status}
Decision: {result.decision}
Processing Time: {result.processing_time:.2f} seconds

COMPANY DESCRIPTION
==================
{result.company_description}

BUSINESS ANALYSIS
================
Business Nature: {result.business_analysis.business_nature}
Customer Type: {result.business_analysis.customer_type}
Operation Type: 
{result.business_analysis.operation_type}
Countries/Regions: {result.business_analysis.countries}
Products & Services: {result.business_analysis.products_services}
Target Customers: {result.business_analysis.target_customers}
Target Industries: {result.business_analysis.target_industries}
Business Model: {result.business_analysis.business_model}
Global Presence: {result.business_analysis.global_presence}

KEYWORD MATCHES
==============
{', '.join(result.keyword_matches) if result.keyword_matches else 'No keywords matched'}

ERROR INFORMATION
================
{result.error_message if result.error_message else 'No errors reported'}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report
    
    def _generate_summary_report(self) -> str:
        """Generate summary report for all analyzed websites"""
        total = len(self.results)
        if total == 0: # Handle case with no results
            return "No websites were analyzed."

        accepted = sum(1 for r in self.results if r.decision == "Accepted")
        
rejected = sum(1 for r in self.results if r.decision.startswith("Rejected"))
        errors = sum(1 for r in self.results if r.status == "Error")
        
        avg_time = sum(r.processing_time for r in self.results) / total if total > 0 else 0
        
        # UX: Provides aggregate statistics for a high-level overview.
        # Analysis by customer type
        customer_types_summary: Dict[str, int] = {} # Typed dictionary
        for result in self.results:
   
         ct = result.business_analysis.customer_type
            customer_types_summary[ct] = customer_types_summary.get(ct, 0) + 1
        
        # Analysis by operation type
        operation_types_summary: Dict[str, int] = {} # Typed dictionary
        for result in self.results:
            ot = result.business_analysis.operation_type
            operation_types_summary[ot] = operation_types_summary.get(ot, 0) + 
1
        
        # UX: Clear structure with headings and percentages for easy consumption of summary data.
        summary = f"""
ENTERPRISE WEBSITE INTELLIGENCE SUMMARY REPORT
==============================================

OVERVIEW
========
Total Websites Analyzed: {total}
Accepted: {accepted} ({accepted/total*100:.1f}% if total > 0 else 0.0}%)
Rejected: {rejected} ({rejected/total*100:.1f}% if total > 0 else 0.0}%)
Errors: {errors} ({errors/total*100:.1f}% if total > 0 else 0.0}%)
Average Processing Time: {avg_time:.2f} seconds

CUSTOMER TYPE DISTRIBUTION
=========================
"""
        for ct, count in customer_types_summary.items():
            summary += f"{ct}: {count} ({count/total*100:.1f}% if total > 0 else 0.0}%)\n"
        
        summary += f"""
OPERATION TYPE DISTRIBUTION
===========================
"""
        for ot, count in operation_types_summary.items():
   
         summary += f"{ot}: {count} ({count/total*100:.1f}% if total > 0 else 0.0}%)\n"
        
        summary += f"""
ACCEPTED WEBSITES
================
"""
        # UX: Lists accepted websites for quick identification.
        accepted_list = [f"- {result.website}: {result.company_description}" for result in self.results if result.decision == "Accepted"]
        summary += "\n".join(accepted_list) if accepted_list else "No websites were accepted."
        
        summary += f"""

REJECTED WEBSITES
================
"""
        # UX: Lists rejected websites along with reasons.
   
     rejected_list = [f"- {result.website}: {result.decision}" for result in self.results if result.decision.startswith("Rejected")]
        summary += "\n".join(rejected_list) if rejected_list else "No websites were rejected."
        
        summary += f"""

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return summary
    
    def _save_raw_data(self) -> Optional[str]: # Return type hint
        """Save raw analysis data as JSON"""
        if not self.results: # Do not create empty JSON if no results
            return None
     
   try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            temp_dir = tempfile.mkdtemp(prefix="raw_analysis_")
            json_filename = os.path.join(temp_dir, f"raw_analysis_data_{timestamp}.json")
            
            # UX: Converts results to a serializable format for raw data export, useful for external processing.
            raw_data = []
           
 for result in self.results:
                raw_data.append({
                    "website": result.website,
                    "status": result.status,
                    "company_description": result.company_description,
                 
   "business_analysis": asdict(result.business_analysis), # Converts dataclass to dict
                    "decision": result.decision,
                    "keyword_matches": result.keyword_matches,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message,
            
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # UX: Saves data in JSON format with indentation for readability.
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False)
            
            return json_filename 
if os.path.exists(json_filename) else None
        except Exception as e:
            logger.error(f"Error saving raw data: {str(e)}")
            return None

# Enhanced Gradio Interface - FIXED VERSION
def create_enterprise_ui():
    """Create the sophisticated enterprise interface with enhanced styling"""
    
    processor_ref = [None] # Using a list to pass by reference for 'nonlocal' behavior in handlers

    # UX: Establishes a professional and modern theme using Gradio's theming capabilities.
    # Primary and secondary hues (indigo, slate) create a corporate feel.
    # GoogleFont("Poppins") enhances readability and modern aesthetics. [cite: 142]
    with gr.Blocks(
        title="Enterprise Website Intelligence Platform",
        theme=gr.themes.Soft(
 
           primary_hue="indigo", # UX: Accent color for key elements.
            secondary_hue="slate",  # UX: Neutral color for secondary elements.
            neutral_hue="stone",    # UX: Base color for backgrounds and general UI.
            font=[gr.themes.GoogleFont("Poppins"), "Inter", "system-ui", "sans-serif"], # UX: Prioritizes modern, legible sans-serif fonts. [cite: 144]
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Monaco", "monospace"] # UX: Clear monospace font for code/log displays.
        ),
        css="""
        /* --- Base & Typography --- */
        /* UX: Global styling for a consistent look and feel. 
           Uses a subtle linear gradient for the background, enhancing visual depth.
           Poppins and Inter fonts are chosen for modern readability. */
body, .gradio-container {
            background: linear-gradient(145deg, #fafbfc 0%, #f1f3f4 50%, #e8eaed 100%) !important; 
min-height: 100vh; /* Ensures background covers the full view height. */
            font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; /* UX: Primary font stack. */ [cite: 144]
        }
        
        /* UX: Consistent font families for headings and paragraph/label text, improving typographic hierarchy. */
h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif !important; }
p, span, div, label { font-family: 'Inter', sans-serif !important; }

        /* --- Header Styling --- */
        /* UX: Creates a prominent and visually appealing header section.
           Uses gradients and borders for a polished, modern look. */
        .enterprise-header {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%); 
            border: 1px solid rgba(99, 102, 241, 0.1); /* Subtle border matching accent. */ 
            border-radius: 20px; /* Soft rounded corners for a modern feel. */
            padding: 3rem 2.5rem; /* Ample padding (48px, 40px) for spacing, following 8px rhythm. */
            margin-bottom: 2.5rem; /* Consistent margin (40px). */
            text-align: center;
box-shadow: 0 10px 35px rgba(99, 102, 241, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05); /* Subtle shadow for depth. */
position: relative; /* For pseudo-elements. */
            overflow: hidden; /* Clips ::before/::after effects. */
        }
        
        /* UX: Adds a colored top border accent to the header, reinforcing branding. */
        .enterprise-header::before {
            content: ''; 
            position: absolute; top: 0; left: 0; right: 0; height: 3px; 
background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 25%, #a855f7 50%, #8b5cf6 75%, #6366f1 100%); /* Accent gradient. */
}
        
        /* UX: Subtle animated radial gradient for a dynamic background effect in the header. */
        .enterprise-header::after {
            content: ''; 
            position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; 
            background: radial-gradient(circle, rgba(99, 102, 241, 0.03) 0%, transparent 70%); 
animation: glow 8s ease-in-out infinite; /* Smooth, slow animation. */
        }
        
        /* UX: Keyframes for the header's background glow animation. */
        @keyframes glow {
0%, 100% { transform: rotate(0deg) scale(1); }
50% { transform: rotate(180deg) scale(1.1); } /* Subtle scaling and rotation. */
        }
        
        /* UX: Styles the main title for strong visual impact.
           Uses a gradient text fill for a premium appearance. */
        .enterprise-title {
            font-size: 3.2rem; /* Large font size for importance. */ 
font-weight: 700; /* Bold weight. */
            color: #1e293b; /* Base color if gradient fails. */
            margin-bottom: 1rem; /* 16px margin. */
            letter-spacing: -1.5px; /* Tightened letter spacing for a modern look. */
            background: linear-gradient(135deg, #1e293b 0%, #475569 30%, #6366f1 70%, #8b5cf6 100%);
            -webkit-background-clip: text; 
-webkit-text-fill-color: transparent; /* Makes text color transparent to show gradient. */
            background-clip: text; /* Standard property. */
            position: relative; z-index: 2; /* Ensures text is above pseudo-elements. */
        }
        
        /* UX: Styles the subtitle for a clear, descriptive secondary message. */
        .enterprise-subtitle {
            font-size: 1.2rem; 
color: #64748b; /* Muted grey for secondary text, good contrast. */
            font-weight: 400; /* Regular weight. */
            line-height: 1.7; /* Improved readability. */
            max-width: 750px; margin: 0 auto; /* Centered and constrained width. */
            position: relative; z-index: 2; 
        }
        
        /* --- Group & Panel Styling --- */
        /* UX: Defines the appearance of content groups (cards/panels).
           Subtle gradients, borders, and shadows create a clean, layered look.
           Consistent padding (32px) and margins (32px) maintain rhythm. */
        .gradio-group {
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important; 
            border: 1px solid rgba(99, 102, 241, 0.08) !important; /* Very subtle border. */ 
            border-radius: 16px !important; /* Rounded corners (16px). */
            padding: 2rem !important; /* 32px padding. */ 
            margin-bottom: 2rem !important; /* 32px margin. */ 
box-shadow: 0 4px 20px rgba(99, 102, 241, 0.06),  0 1px 3px rgba(0, 0, 0, 0.03) !important; /* Soft shadow. */
transition: all 0.3s ease !important; /* Smooth transitions for hover effects. */
        }
        
        /* UX: Adds a subtle lift effect on hover for interactive feedback. */
        .gradio-group:hover {
            transform: translateY(-2px) !important; 
box-shadow: 0 8px 30px rgba(99, 102, 241, 0.1), 0 2px 6px rgba(0, 0, 0, 0.05) !important; /* Enhanced shadow on hover. */ [cite: 163]
}
        
        /* UX: Styles section headers within groups for clear visual hierarchy.
           Gradient text and a bottom border add polish. */
        .gradio-group h3 {
            color: #1e293b !important; 
font-weight: 600 !important; /* Semi-bold weight. */
            font-size: 1.4rem !important; /* Clear heading size. */
            margin-bottom: 1.5rem !important; /* 24px margin. */
            padding-bottom: 0.75rem !important; /* 12px padding. */
            border-bottom: 2px solid #f1f5f9 !important; /* Subtle separator. */
background: linear-gradient(90deg, #6366f1, #8b5cf6); /* Accent gradient for text. */
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; 
}
        
        /* --- Input Styling (Textbox, Dropdown) --- */
        /* UX: Consistent styling for text inputs and dropdowns.
           Rounded corners, subtle borders, and gradients create a modern, clean look.
           Padding (14px) enhances usability. */
        .gradio-textbox, .gradio-dropdown {
            border: 1.5px solid #e2e8f0 !important; 
            border-radius: 12px !important; /* Rounded corners (12px). */ 
            background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%) !important; 
            transition: all 0.3s ease !important; /* Smooth transitions. */
            font-family: 'Inter', sans-serif !important; /* Consistent font. */ 
padding: 0.875rem !important; /* 14px padding. */
        }
        
        /* UX: Provides clear visual feedback on focus for inputs.
           Accent-colored border and shadow improve accessibility and interaction. */
        .gradio-textbox:focus, .gradio-dropdown:focus {
            border-color: #6366f1 !important; /* Accent color border on focus. */ 
box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important; /* Focus ring. */
            background: #ffffff !important; /* Brighter background on focus. */
            transform: translateY(-1px) !important; /* Slight lift effect. */ 
}
        
        /* --- Button Styling --- */
        /* UX: Base styling for all buttons, ensuring consistency.
           Gradient background, rounded corners, and shadow create a modern, clickable appearance.
           Ample padding (16px 32px) for easy interaction. */
        .gradio-button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; /* Default accent gradient. */ 
            border: none !important; 
            border-radius: 12px !important; /* Rounded corners (12px). */ 
            color: white !important; /* High contrast text. */
            font-weight: 600 !important; /* Semi-bold text. */
            font-family: 'Poppins', sans-serif !important; /* Consistent button font. */
            padding: 1rem 2rem !important; /* 16px vertical, 32px horizontal padding. */ 
transition: all 0.3s ease !important; /* Smooth transitions. */
box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25), 0 1px 3px rgba(0, 0, 0, 0.1) !important; /* Default shadow. */
position: relative !important; overflow: hidden !important; /* For hover effect. */
        }
        
        /* UX: Adds a subtle sheen animation on hover for buttons, enhancing interactivity. */
        .gradio-button::before {
            content: ''; 
            position: absolute; top: 0; left: -100%; width: 100%; height: 100%; 
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent); 
transition: left 0.5s ease; /* Smooth animation of the sheen. */
        }
        
        /* UX: Defines button appearance on hover for clear interactive feedback.
           Slightly darker gradient, lift effect, and enhanced shadow. */
        .gradio-button:hover {
            background: linear-gradient(135deg, #5855eb 0%, #7c3aed 100%) !important; /* Darker gradient on hover. */ 
transform: translateY(-2px) !important; /* Lift effect. */
box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35), 0 2px 6px rgba(0, 0, 0, 0.15) !important; /* Stronger shadow. */ [cite: 177]
}
        .gradio-button:hover::before { left: 100%; } /* Animates sheen across the button. */
        
        /* UX: Styles primary action buttons with a distinct color (green for positive action)
           to guide the user towards the main task. */
        .gradio-button.primary { /* Solid button for primary actions. */
            background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important; /* Green gradient for 'Execute'. */ 
box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25), 0 1px 3px rgba(0, 0, 0, 0.1) !important; [cite: 180]
}
        .gradio-button.primary:hover {
            background: linear-gradient(135deg, #047857 0%, #059669 50%, #10b981 100%) !important; /* Darker green on hover. */ 
box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35), 0 2px 6px rgba(0, 0, 0, 0.15) !important; [cite: 182]
}
        
        /* UX: Styles secondary buttons with a more muted appearance (grey)
           to differentiate them from primary actions but maintain a professional look. */
        .gradio-button.secondary { /* Solid, but less prominent, for secondary actions. */
            background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%) !important; /* Grey gradient. */ 
box-shadow: 0 4px 15px rgba(100, 116, 139, 0.2), 0 1px 3px rgba(0, 0, 0, 0.08) !important; [cite: 184]
}
        .gradio-button.secondary:hover {
            background: linear-gradient(135deg, #475569 0%, #64748b 100%) !important; /* Darker grey on hover. */ 
box-shadow: 0 8px 25px rgba(100, 116, 139, 0.3), 0 2px 6px rgba(0, 0, 0, 0.12) !important; [cite: 186]
}

        /* --- DataFrame Styling --- */
        /* UX: Professional styling for data tables.
           Clean background, subtle borders, and clear typography enhance data presentation. */
        .gradio-dataframe {
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important; 
            border: 1px solid rgba(99, 102, 241, 0.1) !important; /* Subtle accent border. */ 
            border-radius: 16px !important; /* Rounded corners (16px). */ 
            overflow: hidden !important; /* Ensures content respects border radius. */
box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05) !important; /* Soft shadow. */ [cite: 189]
}
        .gradio-dataframe table { background: transparent !important; font-family: 'Inter', sans-serif !important; } /* Clear table background. */
        
        /* UX: Styles table headers for clarity and visual distinction. */
        .gradio-dataframe th {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important; /* Light grey gradient for headers. */ 
            color: #1e293b !important; /* Dark text for contrast. */ 
            font-weight: 600 !important; /* Semi-bold. */ 
            font-family: 'Poppins', sans-serif !important; /* Header font. */ 
            border-bottom: 2px solid #e2e8f0 !important; /* Separator. */
            padding: 1rem 0.75rem !important; /* 16px 12px padding. */ 
}
        
        /* UX: Styles table cells for readability. */
        .gradio-dataframe td {
            border-bottom: 1px solid #f1f5f9 !important; /* Light row separator. */ 
padding: 0.875rem 0.75rem !important; /* 14px 12px padding. */
            transition: background-color 0.2s ease !important; /* Smooth hover effect. */
        }
        /* UX: Row hover effect for better data tracking. */
        .gradio-dataframe tr:hover td { background-color: rgba(99, 102, 241, 0.02) !important; }

        /* --- Status Messaging & Log Output --- */
        /* UX: Styles HTML components used for status messages.
           Consistent panel appearance with subtle borders and shadows. */
        .gradio-html { /* Used for status_output */
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important; 
            border: 1px solid rgba(99, 102, 241, 0.1) !important; /* Subtle accent border. */ 
            border-radius: 12px !important; /* Rounded corners (12px). */
            padding: 1.25rem !important; /* 20px padding. */ 
box-shadow: 0 2px 10px rgba(99, 102, 241, 0.05), 0 1px 3px rgba(0, 0, 0, 0.03) !important; /* Light shadow. */
font-family: 'Inter', sans-serif !important; /* Consistent font. */
        }

        /* UX: Styles readonly textboxes used for log output with a dark "console" theme
           for clear differentiation and readability of log messages. */
        .gradio-textbox[readonly] { /* Targets log_output */
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important; /* Dark background. */ 
            border: 1px solid #475569 !important; /* Muted border. */ 
            color: #e2e8f0 !important; /* Light text for contrast. */ 
            font-family: 'JetBrains Mono','Monaco','Menlo','Ubuntu Mono',monospace !important; /* Monospace font for logs. */ 
            font-size: 0.9rem !important; 
line-height: 1.6 !important; /* Improved line spacing for logs. */
        }
        
        /* --- File Upload Styling --- */
        /* UX: Enhances the appearance of the file upload component.
           Dashed border and subtle background provide a clear dropzone indication. */
        .gradio-file {
            background: linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%) !important; 
            border: 2px dashed #cbd5e1 !important; /* Dashed border to indicate dropzone. */ 
            border-radius: 12px !important; /* Rounded corners (12px). */
            transition: all 0.3s ease !important; /* Smooth transitions. */ 
}
        /* UX: Visual feedback on hover for the file upload area. */
        .gradio-file:hover {
            border-color: #6366f1 !important; /* Accent color border on hover. */ 
background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%) !important; /* Lighter background on hover. */
            transform: translateY(-1px) !important; /* Slight lift. */ 
}
        
        /* --- Accordion Styling --- */
        /* UX: Styles accordion elements for collapsible content sections.
           Clean, modern appearance that integrates well with group styling. */
        .gradio-accordion {
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important; 
            border: 1px solid rgba(99, 102, 241, 0.08) !important; /* Subtle border. */ 
            border-radius: 12px !important; /* Rounded corners (12px). */
            overflow: hidden !important; /* Clips content within rounded corners. */
box-shadow: 0 2px 8px rgba(99, 102, 241, 0.04) !important; /* Very light shadow. */
        }
        /* UX: Styles accordion headers for clear clickability and visual separation. */
        .gradio-accordion summary {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important; /* Light header background. */ 
            color: #1e293b !important; /* Dark text. */ 
            font-weight: 500 !important; /* Medium weight. */ 
            font-family: 'Poppins', sans-serif !important; /* Consistent header font. */
            padding: 1.25rem !important; /* 20px padding. */
            cursor: pointer !important; /* Indicates interactivity. */ 
transition: all 0.2s ease !important; /* Smooth hover transition. */
        }
        .gradio-accordion summary:hover { background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important; } /* Darker on hover. */
        
        /* --- Checkbox Group Styling --- */
        /* UX: Styles checkbox group labels to appear as toggle buttons or tags.
           Improves visual appeal and interactivity of selection. */
        .gradio-checkboxgroup label {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important; /* Light background for unselected. */ 
            border: 1.5px solid #e2e8f0 !important; 
            border-radius: 10px !important; /* Rounded corners (10px). */
            padding: 0.75rem 1rem !important; /* 12px 16px padding. */
            margin: 0.375rem !important; /* 6px margin around each. */
            transition: all 0.3s ease !important; /* Smooth transitions. */ 
font-family: 'Inter', sans-serif !important; font-weight: 500 !important; /* Text styling. */
        }
        /* UX: Hover effect for checkbox labels. */
        .gradio-checkboxgroup label:hover {
            background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%) !important; /* Accent-based hover. */ 
            border-color: #6366f1 !important; /* Accent border. */ 
            transform: translateY(-1px) !important; /* Lift effect. */
box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15) !important; /* Subtle shadow. */ [cite: 210]
}
        /* UX: Styles selected checkboxes with accent color for clear visual indication. */
        .gradio-checkboxgroup input:checked + span { /* This targets the span inside the label for Gradio */
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; /* Accent gradient for selected. */ 
            color: white !important; /* White text for contrast on accent background. */ 
            /* To make the whole label background change, Gradio's structure might need specific targeting.
               If the above doesn't make the entire label background change, it's a limitation of simple CSS over Gradio's output.
               The intention is for the selected item to be visually distinct using the accent color. */
        }
                
        /* --- Responsive Enhancements --- */
        /* UX: Adjusts layout and font sizes for smaller screens (e.g., tablets and mobiles)
           to ensure usability and maintain a good visual hierarchy. */
        @media (max-width: 768px) {
            .enterprise-title { font-size: 2.5rem !important; } /* Reduced title size. */
            .enterprise-subtitle { font-size: 1.1rem !important; } /* Reduced subtitle size. */
            .gradio-group { padding: 1.5rem !important; } /* Reduced group padding (24px). */
            .gradio-button { padding: 0.875rem 1.5rem !important; } /* Reduced button padding (14px 24px). */
        }
        
        /* --- Loading & Status States --- */
        /* UX: Shimmer animation for loading states, providing visual feedback that content is being prepared. */
        @keyframes shimmer {
0% { transform: translateX(-100%); }
100% { transform: translateX(100%); }
        }
        .loading { position: relative; overflow: hidden; } /* Container for shimmer effect. */
        .loading::after { /* The shimmer element itself. */
            content: ''; 
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent); /* Subtle accent shimmer. */ 
animation: shimmer 2s infinite; /* Continuous animation. */
        }
        
        /* UX: Distinct visual styles for different status messages (success, error, processing)
           using color-coding for quick comprehension. Adheres to WCAG AA by using distinct background/border and text colors. */
        .status-success { /* Applied to status_output via JS if needed, or use Gradio's capabilities. */
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important; /* Greenish background. */ 
            border: 1px solid #10b981 !important; /* Green border. */ 
            color: #047857 !important; /* Dark green text for contrast. */ 
        }
        .status-error {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%) !important; /* Reddish background. */ 
            border: 1px solid #ef4444 !important; /* Red border. */ 
            color: #dc2626 !important; /* Dark red text for contrast. */ 
        }
        .status-processing {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important; /* Bluish background. */ 
            border: 1px solid #3b82f6 !important; /* Blue border. */ 
            color: #1d4ed8 !important; /* Dark blue text for contrast. */ 
        }
        
        /* --- Elegant Shadows (Utility) --- */
        /* UX: Utility classes for applying consistent, elegant shadows if needed directly on elements. */
        .shadow-elegant {
box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }
        .shadow-elegant-hover:hover {
box-shadow: 0 8px 30px rgba(99, 102, 241, 0.12), 0 2px 6px rgba(0, 0, 0, 0.08) !important;
        }
        """
    ) as app:
        
        # UX: The main header sets a professional tone and clearly states the platform's purpose.
        # Emojis add a touch of visual appeal without being unprofessional.
        gr.HTML("""
            <div class="enterprise-header">
                <h1 class="enterprise-title">🎯 Enterprise Website Intelligence</h1>
                <p class="enterprise-subtitle">Advanced AI-powered business intelligence platform • Automated website analysis • 
Strategic insights & competitive intelligence</p>
            </div>
        """)
        
        # State variables
        csv_file_state = gr.State(None) # UX: Holds the path to the generated CSV for download.
        all_results_df_state = gr.State(pd.DataFrame())  # UX: Stores the accumulated results DataFrame for display and download.

        # Input Section
        # UX: Groups related input controls within a styled "card" for better organization.
        # The section header "Intelligence Configuration" clearly defines its purpose. [cite: 231]
        with gr.Group():
    
        gr.HTML("<h3>⚙️ Intelligence Configuration</h3>") # UX: Section title using custom-styled h3.
            
            with gr.Row(): # UX: Arranges URL input and file upload side-by-side for efficient use of space.
                with gr.Column(scale=3): # UX: Allocates more space to the primary URL input.
                    urls_input = gr.Textbox(
                        label="🌐 Target 
Websites", # UX: Clear label with emoji for quick recognition.
                        placeholder="example.com\ncompany.org\nbusiness.net", # UX: Example input format.
                        lines=4,
                        # UX: Helper text guides the user on how to input multiple URLs.
                        info="Enter one URL per line for comprehensive analysis"
                    )
 
               with gr.Column(scale=1): # UX: Smaller column for the secondary file upload action.
                    file_upload = gr.File(
                        label="📄 Import URL List", # UX: Clear label for file import.
                        file_types=[".txt", ".csv"], # UX: Specifies allowed file types, improving usability.
                  elem_classes=["shadow-elegant-hover"] # UX: Applies a subtle hover shadow for interactivity.
                    )
                    # UX: Button to trigger loading URLs from the uploaded file.
                    # `size="sm"` makes it less prominent than the main analysis button.
                    load_button = gr.Button("📥 Load File", size="sm", elem_classes=["secondary"]) # UX: Secondary style for non-primary action.

            with gr.Row(): # UX: Groups related filter inputs.
                business_terms_input = gr.Textbox(
                    label="🔍 Strategic 
Keywords", # UX: Clear label for keyword input.
                    placeholder="technology, manufacturing, consulting, finance", # UX: Example input.
                    # UX: Helper text clarifies the purpose of these keywords.
                    info="Define business intelligence filters (comma-separated)"
                )
                api_key_input = gr.Textbox(
                    
label="🔑 AI API Key", # UX: Clear label, using a key emoji. Changed from lock for clarity on "API Key".
                    placeholder="sk-...", # UX: Standard placeholder for API keys.
                    type="password", # UX: Masks API key input for security.
                    # UX: Helper text indicates the necessity of the API key.
                    info="Required for advanced AI processing"
                )
            
            # UX: Groups checkbox filters for market segments and business models.
            # `CheckboxGroup` provides a user-friendly way to select multiple options.
       with gr.Row():
                customer_type_input = gr.CheckboxGroup(
                    choices=[bt.value for bt in BusinessType], # UX: Dynamically populates choices from Enum for maintainability.
                    label="👥 Market Segments", # UX: Clear label.
                    # UX: Helper text explains the filter's purpose.
                    info="Filter by customer engagement model"
  
              )
                operation_type_input = gr.CheckboxGroup(
                    choices=[ot.value for ot in OperationType], # UX: Dynamically populates choices from Enum.
                    label="🏭 Business Models", # UX: Clear label.
                    # UX: Helper text explains the filter's purpose.
                    info="Filter by operational structure"
 
               )
            
            # UX: Primary call-to-action button.
            # `variant="primary"` applies the distinct primary button styling (green).
            # `size="lg"` makes it prominent.
            submit_button = gr.Button(
                "🚀 Execute Analysis",
                variant="primary", # UX: Clearly indicates the main action button. [cite: 179, 240]
                size="lg" # UX: Larger size for emphasis. [cite: 240]
         
   )
        
        # Status and Results
        # UX: Provides a dedicated area for status updates, styled by `.gradio-html`.
        status_output = gr.HTML("🎯 Ready to deploy enterprise intelligence analysis") # Initial status message.
        
        # UX: Groups results display and logs within a styled "card".
        # "Intelligence Dashboard" clearly titles this output section.
        with gr.Group():
            gr.HTML("<h3>📊 Intelligence Dashboard</h3>") # UX: Section title.
            results_output = gr.DataFrame(
                interactive=False, # UX: Results are for display, not interaction.
 
               wrap=False, # UX: Prevents text wrapping in cells for cleaner table layout.
                height=600 # UX: Fixed height ensures consistent layout, with scrolling for more data.
            )
            
            # UX: Execution log is placed within an accordion, keeping the UI clean by default.
            # `open=False` means it's collapsed initially.
            with gr.Accordion("📋 Execution Log", open=False):
                log_output = gr.Textbox(
           
         lines=10, # UX: Sufficient lines for viewing recent logs.
                    max_lines=20, # UX: Limits excessive growth.
                    interactive=False, # UX: Log is read-only.
                    label="Execution Log" # Accessibility: Adds a label.
                )
        
        # Download Section
        # UX: Groups download buttons for reports within a styled "card".
        # "Intelligence Reports" clearly titles this section. [cite: 243]
        with gr.Group():
        
    gr.HTML("<h3>📦 Intelligence Reports</h3>") # UX: Section title.
            
            # UX: Arranges download buttons in a row for a compact layout.
            # `variant="secondary"` applies secondary button styling (grey) for these actions.
            with gr.Row():
                download_csv_button = gr.Button("📊 Export Data (CSV)", variant="secondary") # UX: Clear label for CSV export.
                download_reports_button = gr.Button("📄 Generate Reports (ZIP)", variant="secondary") # UX: Clear label for ZIP reports.
                download_raw_button = gr.Button("🗂️ Raw Intelligence (JSON)", variant="secondary") # UX: Clear label for JSON export.
      
      
            # UX: Hidden File components are used to trigger browser downloads.
            # `visible=False` keeps them out of the layout until a file is ready.
            with gr.Row():
                csv_download_file = gr.File(label="CSV Data Export", visible=False, interactive=False)
                reports_download_file = gr.File(label="Detailed Reports ZIP", visible=False, interactive=False)
                raw_download_file = gr.File(label="Raw JSON Data", visible=False, interactive=False)
        
        # Processing function for the UI

        def process_websites_ui_wrapper(urls_text, business_terms_text, api_key, customer_types, operation_types, progress=gr.Progress(track_tqdm=True)):
            nonlocal processor_ref # Use the list to modify the outer scope variable
            
            # UX: Provides immediate feedback if essential inputs are missing.
            if not urls_text or not urls_text.strip():
                yield "❌ No URLs provided. Please enter target websites.", pd.DataFrame(), None, "Error: No URLs provided.", all_results_df_state.value
                return
    
            
            if not api_key or not api_key.strip():
                yield "🔑 API key required. Please provide an AI API key.", pd.DataFrame(), None, "Error: API key missing.", all_results_df_state.value
                return
            
            urls = [url.strip() for url in 
urls_text.split('\n') if url.strip()]
            if not urls:
                yield "❌ URL list is empty or contains only whitespace.", pd.DataFrame(), None, "Error: No valid URLs found after parsing.", all_results_df_state.value
                return

            business_terms = [term.strip() for term in (business_terms_text or "").split(',') if term.strip()]
            
            # Initialize processor (this will clear previous results internally)
            processor_ref[0] = StreamingWebsiteProcessor(api_key, max_workers=4)
            
            # UX: Iterates through streamed results, updating UI progressively.
            # This provides real-time feedback to the user during long operations.
            log_accumulator = ""
            total_urls = len(urls)
            
            # Initialize all_results_df for this run
            current_run_results_list = []

            for i, stream_result in enumerate(processor_ref[0].process_websites_stream(
     
           urls, business_terms, customer_types, operation_types
            )):
                progress(i / total_urls, desc=f"Processing URL {i+1}/{total_urls}")

                status, _, csv_file_path, log_message, is_final = stream_result
                
                if log_message: # Accumulate log messages
                    log_accumulator += f"{time.strftime('%H:%M:%S')} - {log_message}\n"
                
                # The StreamingWebsiteProcessor now manages the full list of results internally.
                # We get the complete current dataframe from it.
                current_complete_df = processor_ref[0]._results_to_dataframe()
                current_run_results_list = processor_ref[0].results # Keep a reference to the raw results list

                yield status, current_complete_df, csv_file_path, log_accumulator, current_complete_df # Update all_results_df_state
                
                if is_final:
                    csv_file_state.value = csv_file_path # Update the shared state for download
                    all_results_df_state.value = current_complete_df # Persist final dataframe state

            # Update the global state for downloads after the stream is complete
            # This uses the processor instance that did the work.
            if processor_ref[0]:
                all_results_df_state.value = processor_ref[0]._results_to_dataframe()


        def load_urls_from_file_py(file_obj):
            # UX: Handles loading URLs from a text or CSV file.
            if file_obj is None:
                return "" # Return empty string if no file
            try:
                # Gradio File component provides a file-like object (tempfile._TemporaryFileWrapper)
                with open(file_obj.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                # UX: Provides feedback in the log if file reading fails. [cite: 252]
                log_accumulator.value += f"{time.strftime('%H:%M:%S')} - URLs loaded from file: {file_obj.name}\n"
                return content
            except Exception as e: [cite: 252]
                error_msg = f"Error reading file {file_obj.name if hasattr(file_obj, 'name') else 'unknown file'}: {str(e)}"
                logger.error(error_msg)
                # Also update log_output on UI
                log_accumulator.value += f"{time.strftime('%H:%M:%S')} - {error_msg}\n"
                # Optionally, raise a Gradio error to show it more prominently
                gr.Error(f"Failed to read file: {str(e)}")
                return "" # Return empty string or current content on error
        
        # Download Handlers
        # UX: These functions handle the logic for preparing and triggering file downloads.
        # They update the visibility of hidden `gr.File` components to make downloads available.

        def handle_csv_download_py(current_results_df):
            # UX: Generates a CSV file from the current results for download. [cite: 253]
            if current_results_df is not None and not current_results_df.empty:
                try:
                    processor = processor_ref[0]
                    if processor: # Use the existing processor which has the results and save method
                        csv_path = processor._save_results_to_csv()
                        if csv_path:
                             log_accumulator.value += f"{time.strftime('%H:%M:%S')} - CSV export ready: {csv_path}\n"
                             return gr.update(value=csv_path, visible=True)
                    # Fallback if processor not available or failed, try to save DataFrame directly
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    temp_dir = tempfile.mkdtemp(prefix="fallback_csv_")
                    csv_filename = os.path.join(temp_dir, f"enterprise_analysis_fallback_{timestamp}.csv")
                    current_results_df.to_csv(csv_filename, index=False, escapechar='\\', quoting=1)
                    log_accumulator.value += f"{time.strftime('%H:%M:%S')} - Fallback CSV export ready: {csv_filename}\n"
                    return gr.update(value=csv_filename, visible=True) # UX: Makes the download link visible. [cite: 254]
                except Exception as e:
                    logger.error(f"Error creating CSV for download: {str(e)}")
                    log_accumulator.value += f"{time.strftime('%H:%M:%S')} - Error creating CSV: {str(e)}\n"
                    gr.Error(f"Could not prepare CSV: {str(e)}")
            log_accumulator.value += f"{time.strftime('%H:%M:%S')} - No data to export for CSV.\n"
            return gr.update(visible=False) # UX: Keeps download link hidden if no data.

        def handle_reports_download_py():
            # UX: Generates a ZIP file containing detailed reports. [cite: 255]
            processor = processor_ref[0]
            if processor and processor.results:
                try:
                    reports_file = processor._save_detailed_reports()
                    if reports_file:
                        log_accumulator.value += f"{time.strftime('%H:%M:%S')} - Reports ZIP ready: {reports_file}\n"
                        return gr.update(value=reports_file, visible=True) # UX: Makes download link visible. [cite: 256]
                except Exception as e:
                    logger.error(f"Error generating reports for download: {str(e)}")
                    log_accumulator.value += f"{time.strftime('%H:%M:%S')} - Error generating reports: {str(e)}\n"
                    gr.Error(f"Could not prepare reports ZIP: {str(e)}")
            log_accumulator.value += f"{time.strftime('%H:%M:%S')} - No results available to generate reports.\n"
            return gr.update(visible=False)

        def handle_raw_data_download_py():
            # UX: Generates a JSON file with raw analysis data. [cite: 257]
            processor = processor_ref[0]
            if processor and processor.results:
                try:
                    raw_file = processor._save_raw_data()
                    if raw_file:
                        log_accumulator.value += f"{time.strftime('%H:%M:%S')} - Raw JSON data ready: {raw_file}\n"
                        return gr.update(value=raw_file, visible=True) # UX: Makes download link visible.
                except Exception as e: [cite: 258]
                    logger.error(f"Error generating raw data for download: {str(e)}")
                    log_accumulator.value += f"{time.strftime('%H:%M:%S')} - Error generating raw JSON: {str(e)}\n"
                    gr.Error(f"Could not prepare raw JSON data: {str(e)}")
            log_accumulator.value += f"{time.strftime('%H:%M:%S')} - No results available to generate raw data.\n"
            return gr.update(visible=False)

        # Event handlers
        # UX: Connects UI elements (buttons) to their respective Python functions.
        # `submit_button.click` triggers the main analysis process.
        # Outputs are mapped to UI components to display results and status updates.
        
        # Create a shared state for log accumulation
        log_accumulator = gr.State("")

        submit_button.click(
            fn=process_websites_ui_wrapper,
           
 inputs=[urls_input, business_terms_input, api_key_input, customer_type_input, operation_type_input],
            # Order of outputs must match the yield statement in process_websites_ui_wrapper
            outputs=[status_output, results_output, csv_file_state, log_output, all_results_df_state]
        )
        
        # UX: `load_button.click` triggers loading URLs from the selected file into the textbox.
        load_button.click(
            fn=load_urls_from_file_py,
            inputs=[file_upload],
            outputs=[urls_input]
        )
        
        # UX: Download buttons trigger their respective file generation and download handlers.

        download_csv_button.click(
            fn=handle_csv_download_py,
            inputs=[all_results_df_state], # Pass the DataFrame state
            outputs=[csv_download_file]
        )
        
        download_reports_button.click(
            fn=handle_reports_download_py,
          
  inputs=[], # No direct inputs, uses processor_ref
            outputs=[reports_download_file]
        )
        
        download_raw_button.click(
            fn=handle_raw_data_download_py,
            inputs=[], # No direct inputs, uses processor_ref
            outputs=[raw_download_file]
        )
    
    return app

# Main execution
if __name__ == "__main__":
    
print("🎯 Starting Enterprise Website Intelligence Platform")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = []
    
    try:
        import langchain
        print("✅ Core AI framework (LangChain) available")
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import openai # Indirectly via langchain_openai
        print("✅ AI processing engine (OpenAI) available via LangChain")
  
  except ImportError: # This check might be redundant if langchain_openai is the primary import
        missing_deps.append("openai or langchain-openai")
    
    if LANGGRAPH_AVAILABLE:
        print("✅ Advanced workflow engine (LangGraph) enabled")
    else:
        print("⚠️  Advanced workflows (LangGraph) not available - Using standard processing.")
        print("   For enhanced features like stateful graph execution, install LangGraph: pip install langgraph")
    
    try:
        import gradio
        print("✅ UI framework (Gradio) available")
    except ImportError:
        missing_deps.append("gradio")

    try:
        import pandas
        print("✅ Data handling library (Pandas) available")
    except ImportError:
        missing_deps.append("pandas")
        
    try:
        import beautifulsoup4
        print("✅ HTML parsing library (BeautifulSoup4) available")
    except ImportError:
        missing_deps.append("beautifulsoup4")

    try:
        import requests
        print("✅ HTTP request library (Requests) available")
    except ImportError:
        missing_deps.append("requests")


    if missing_deps:
        print(f"\n❌ Missing critical dependencies: {', '.join(missing_deps)}")
        print("Please install them to ensure full functionality.")
        print("Recommended base installation: pip install langchain langchain-openai gradio requests beautifulsoup4 pandas")
# Basic installation command from the file header
        if "langgraph" not in missing_deps and not LANGGRAPH_AVAILABLE and "LangGraph" not in [dep for dep in missing_deps]: # if langgraph is not already listed as missing
             print("Optional for advanced workflows: pip install langgraph")
        exit(1)
    
    print("\n🌐 Initializing enterprise interface...")
    print("=" * 60)
    
    # Test basic functionality (optional, can be removed for production)
    try:
        # Test with a dummy key for initialization purposes only.
        # Real operations will require a valid API key provided in the UI.
        test_analyzer = AdvancedWebsiteAnalyzer("test-key-will-not-work")
        print("✅ Intelligence engine initialization successful (dummy key).")
    except Exception as e:
        print(f"⚠️  Intelligence engine initialization warning: {e}")
        print("   This might be due to dummy API key or other configuration issues not affecting UI launch.")
    
    app = create_enterprise_ui()
    print("\n🚀 Launching Gradio application...")
    print("   Access the platform via your browser, typically at: http://0.0.0.0:7862 or http://127.0.0.1:7862")
    app.launch(
 
       server_name="0.0.0.0", # Makes it accessible on the network
        server_port=7862,
        share=False, # Set to True to create a public link (requires internet)
        inbrowser=True, # Attempts to open in browser automatically
        show_error=True # Shows Python errors in the browser for easier debugging during development
    )

# --- README Section ---
"""
====================================
Enterprise Website Intelligence Platform
====================================

This application provides an AI-powered platform for analyzing websites to extract business intelligence. 
It uses LangChain for its core AI capabilities and Gradio for the user interface.

--------------------
Running the Application
--------------------

1.  **Ensure Python is installed:**
    This application requires Python 3.8 or newer.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```

3.  **Install Dependencies:**
    The application requires several Python libraries. Install them using pip:

    ```bash
    pip install langchain langchain-openai gradio requests beautifulsoup4 pandas
    ```

    For optional advanced workflow capabilities using LangGraph (which provides more robust state management and complex flow construction):
    ```bash
    pip install langgraph
    ```
    If LangGraph is not installed, the application will fall back to a standard sequential processing mode.

4.  **Set OpenAI API Key:**
    The application uses OpenAI models for analysis. You will need a valid OpenAI API key.
    This key is entered directly into the application's UI when you run it.
    Alternatively, for development, you could set it as an environment variable `OPENAI_API_KEY`,
    though the UI input will typically override this for the `AdvancedWebsiteAnalyzer`.

5.  **Run the Python Script:**
    Save the code as a Python file (e.g., `enterprise_analyzer.py`) and run it from your terminal:
    ```bash
    python enterprise_analyzer.py
    ```

6.  **Access the UI:**
    Once the script is running, it will typically print a local URL to the console (e.g., `http://127.0.0.1:7862` or `http://0.0.0.0:7862`).
    Open this URL in your web browser to use the application. The application should also attempt to open in your browser automatically.

--------------------
Using the Platform
--------------------

1.  **Enter Target Websites:** Input URLs one per line in the textbox, or upload a `.txt` or `.csv` file containing a list of URLs.
2.  **Provide AI API Key:** Enter your OpenAI API key. This is required for the analysis.
3.  **Set Filters (Optional):**
    * **Strategic Keywords:** Define comma-separated keywords to look for in website content.
    * **Market Segments:** Filter by B2B, B2C, or Both.
    * **Business Models:** Filter by Manufacturing, Trading, or Services.
4.  **Execute Analysis:** Click the "Execute Analysis" button.
5.  **View Results:** Analysis results will appear in the "Intelligence Dashboard" table.
6.  **Check Logs:** The "Execution Log" accordion provides detailed processing messages.
7.  **Download Reports:** Use the buttons in the "Intelligence Reports" section to download:
    * **CSV Data:** A CSV file of the main analysis results.
    * **Detailed Reports (ZIP):** A ZIP file containing individual text reports for each website and a summary report.
    * **Raw Intelligence (JSON):** A JSON file with the complete raw data from the analysis.

--------------------
Code Structure
--------------------
- **Data Models (`WebsiteData`, `BusinessAnalysis`, etc.):** Define the structure for data handling.
- **`AdvancedWebsiteAnalyzer` Class:** Contains the core logic for website scraping, content extraction, and AI-powered analysis using LangChain and LangGraph (if available).
- **`StreamingWebsiteProcessor` Class:** Manages the concurrent processing of multiple websites and streams results.
- **`create_enterprise_ui()` Function:** Builds the Gradio user interface, including all layouts, components, styling (CSS), and event handlers.
- **Helper Functions:** Various utility functions for URL formatting, text extraction, parsing, etc.

The UI is designed to be professional and "enterprise-grade," with a focus on clarity, usability, and modern aesthetics.
"""
