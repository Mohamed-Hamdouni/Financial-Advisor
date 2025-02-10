import os
import glob
from typing import List, Optional, Tuple

# PDF extraction
try:
    from PyPDF2 import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# LangChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI

import feedparser
import openai

# Date parsing
from datetime import datetime
from dateparser.search import search_dates


import os
import glob
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
# Pour l'extraction de PDF
try:
    from PyPDF2 import PdfReader  # PyPDF2 (recommandé)
except ImportError:
    from PyPDF2 import PdfReader  # fallback

# LangChain - pour la structure "Document", les embeddings, etc.
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.vectorstores import Chroma

# Pour créer des Tools et Agents
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

from langchain_community.document_loaders import PyPDFLoader


# Exemple d'API finance
import yfinance as yf
# Paramètres généraux
CHUNK_SIZE = 1000   # Nombre de caractères par chunk
CHUNK_OVERLAP = 100 # Overlap entre deux chunks pour éviter la coupure en plein milieu de phrase


# 2. Récupérer la liste des PDFs à indexer
PDF_FOLDER = "./data"
VECTOR_DB = "./vector_db" # ou ca: "./chroma_db"
pdf_files = glob.glob(f"{PDF_FOLDER}/*.pdf")





# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)
###############################################################################
# A) Building/Loading the Vector DB (RAG)
###############################################################################

def load_financial_docs_from_pdfs(pdf_folder: str) -> List[Document]:
    """Reads all PDFs in pdf_folder, returns a list of Documents."""
    pdf_files = glob.glob(f"{pdf_folder}/*.pdf")
    docs = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text
        metadata = {"source": pdf_file}
        docs.append(Document(page_content=full_text, metadata=metadata))
    return docs


def build_or_load_vector_db(pdf_folder: str, persist_dir: str) -> Chroma:
    """
    Creates or loads a Chroma DB from the PDF folder.
    If persist_dir exists, loads it; otherwise, builds a new one.
    """
    if os.path.exists(persist_dir):
        print("[INFO] Loading existing Chroma DB...")
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY", ""))
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        return vector_db
    else:
        print("[INFO] Building a new Chroma DB from PDFs...")
        docs = load_financial_docs_from_pdfs(pdf_folder)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY", ""))
        vector_db = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vector_db.persist()
        return vector_db


###############################################################################
# B) Date Parsing in the User Question
###############################################################################

def parse_date_range(question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Tries to detect date references and returns appropriate date ranges:
    - For a specific month: returns first and last day of that month
    - For a specific year: returns first and last day of that year
    - For explicit ranges: returns the range as specified
    Returns (start_date_str, end_date_str) in the format 'YYYY-MM-DD'.
    """
    # Configuration pour dateparser
    settings = {
        'PREFER_DAY_OF_MONTH': 'first',
        'PREFER_DATES_FROM': 'current_period',
        'RELATIVE_BASE': datetime.now(),
        'STRICT_PARSING': False,
        'DATE_ORDER': 'YMD'
    }

    # Prétraitement du texte
    patterns = [
        (r'between\s+(.*?)\s+and\s+(.*?)(?:\?|$)', r'\1, \2'),
        (r'from\s+(.*?)\s+to\s+(.*?)(?:\?|$)', r'\1, \2'),
        (r'in\s+(.*?)(?:\?|$)', r'\1'),
    ]
    
    clean_question = question
    for pattern, replacement in patterns:
        import re
        clean_question = re.sub(pattern, replacement, clean_question)
    
    print(f"Debug - Cleaned question: {clean_question}")

    # Liste des mois pour la détection
    months = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    
    # Recherche des dates
    found = search_dates(clean_question, settings=settings)
    if not found:
        settings['STRICT_PARSING'] = False
        found = search_dates(clean_question, settings=settings)
        if not found:
            date_parts = [part.strip() for part in clean_question.split(',')]
            found = []
            for part in date_parts:
                result = search_dates(part, settings=settings)
                if result:
                    found.extend(result)
    
    print(f"Debug - Raw dates found: {found}")
    
    if not found:
        return (None, None)

    # Si on a trouvé une seule date
    if len(found) == 1:
        text, date_obj = found[0]
        text_lower = text.lower()
        
        # Vérifier si c'est un mois spécifique
        if any(month in text_lower for month in months):
            import calendar
            # Obtenir le dernier jour du mois
            last_day = calendar.monthrange(date_obj.year, date_obj.month)[1]
            start_date = datetime(date_obj.year, date_obj.month, 1)
            end_date = datetime(date_obj.year, date_obj.month, last_day)
            return (
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        # Vérifier si c'est une année complète
        elif str(date_obj.year) in text:
            start_date = datetime(date_obj.year, 1, 1)
            end_date = datetime(date_obj.year, 12, 31)
            return (
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
    
    # Pour les autres cas (plages de dates explicites)
    dates = [date_obj for _, date_obj in found]
    dates.sort()
    
    # Conversion en format string YYYY-MM-DD
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    print(f"Debug - Processed dates: {date_strs}")
    
    if len(date_strs) >= 2:
        return (date_strs[0], date_strs[1])
    elif len(date_strs) == 1:
        return (date_strs[0], None)
    else:
        return (None, None)


###############################################################################
# C) Multi-Query Augmentation + Advanced Retrieval
###############################################################################

from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever

def search_vector_db_advanced(query: str, vector_db, k: int = 3) -> str:
    """
    1) Retrieves top-k relevant docs using the ChromaDB collection directly
    2) Summarizes them in English via GPT
    """
    try:
        # Get embeddings for the query using the embeddings class directly
        results = vector_db.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        all_content = ""
        for idx, doc in enumerate(results['documents'][0], start=1):
            all_content += f"\n--- Excerpt #{idx} ---\n"
            all_content += doc

        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        prompt = f"""
        USER QUESTION: {query}

        Below are EXCERPTS from the knowledge base:
        {all_content}

        Please provide a concise English SUMMARY focusing on the key points 
        that help address the user's question. Do not fabricate any information.
        """
        summary = llm.predict(prompt)
        return summary
        
    except Exception as e:
        print(f"[ERROR] Error in search_vector_db_advanced: {str(e)}")
        raise


###############################################################################
# C1) Finance data with optional date range
###############################################################################

import yfinance as yf

def fetch_market_data_with_range(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    If start_date/end_date are provided (YYYY-MM-DD), fetch historical data from that range.
    Otherwise, fetch current info (regularMarketPrice, marketCap).
    """
    clean_ticker = ticker.replace("'", "").replace('"', "").strip()
    
    try:
        stock = yf.Ticker(clean_ticker)
        
        if start_date and end_date:
            # Get historical data
            try:
                hist = stock.history(start=start_date, end=end_date)
            except Exception as e:
                return f"Error fetching historical data for {clean_ticker}: {e}"
            if hist.empty:
                return (f"No market data found for {clean_ticker} "
                        f"between {start_date} and {end_date}.")
            
            # Example: we can get average closing price or something
            avg_close = hist['Close'].mean()
            min_date = hist.index.min().strftime("%Y-%m-%d")
            max_date = hist.index.max().strftime("%Y-%m-%d")
            return (
                f"Historical market data for {clean_ticker} from {start_date} to {end_date}:\n"
                f"Data range actually found: {min_date} to {max_date}\n"
                f"Average closing price: {avg_close:.2f}\n"
                f"(Total days: {len(hist)})\n"
            )
        else:
            # Current info
            info = stock.info
            long_name = info.get("longName", "Unknown")
            current_price = info.get("regularMarketPrice", "N/A")
            market_cap = info.get("marketCap", "N/A")
            return (
                f"Current market data for {clean_ticker} - {long_name}:\n"
                f"Price: {current_price}\n"
                f"Market cap: {market_cap}\n"
            )
    except Exception as e:
        return f"Error fetching market data for {clean_ticker}: {e}"


###############################################################################
# D) Agents
###############################################################################

# 1) Retrieval Agent (ADVANCED)
def make_retrieval_agent(llm, vector_db: Chroma):
    """
    Instead of calling `search_vector_db_with_summary`, we use `search_vector_db_advanced`
    which implements multi-query augmentation & advanced retrieval.
    """
    def retrieval_tool_func(q: str) -> str:
        return search_vector_db_advanced(q, vector_db, k=4)

    search_tool = Tool(
        name="search_vector_db",
        func=retrieval_tool_func,
        description="Searches internal financial documents with a multi-query approach, returning a concise English summary."
    )
    return initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )


# 2) Grader
grader_prompt = """
You are a financial examiner. You received the following CONTENT from the internal DB.
Reply with "YES" if this information is SUFFICIENT to make an investment decision.
Reply with "NO" if it is NOT sufficient.

CONTENT:
{retrieved_text}
"""

def grader_agent(llm, retrieved_text: str) -> bool:
    response = llm.predict(grader_prompt.format(retrieved_text=retrieved_text))
    return response.strip().upper().startswith("YES")


# 3) Finance Agent
def make_finance_agent(llm, start_date: Optional[str], end_date: Optional[str]):
    def finance_tool_func(ticker: str) -> str:
        return fetch_market_data_with_range(ticker, start_date, end_date)

    finance_tool = Tool(
        name="fetch_finance_data",
        func=finance_tool_func,
        description=(
            "Use this tool to retrieve the last available market data for a given ticker. "
            "If a start/end date was provided, the data is historical for that range; otherwise it's current data."
        )
    )
    return initialize_agent(
        tools=[finance_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )



# 5) Recommender Agent
recommender_prompt = """
You are a professional investment advisor.

Here is the INTERNAL INFORMATION:
{internal_info}

Here is the EXTERNAL INFORMATION:
{external_info}

Based on all the information, please provide a final investment recommendation in English.
Explain your reasoning briefly.
"""

def recommender_agent(llm, internal_info: str, external_info: str) -> str:
    prompt = recommender_prompt.format(
        internal_info=internal_info,
        external_info=external_info
    )
    return llm.predict(prompt)


###############################################################################
# E) Orchestration
###############################################################################

def multi_agent_investment_pipeline(question: str, ticker: str, vector_db: Chroma) -> str:
    try:
        print("[INFO] Starting multi-agent pipeline...")

        # 1) Parse date range
        print("[INFO] Parsing date range...")
        start_date, end_date = parse_date_range(question)
        print(f"[INFO] Date range: {start_date} - {end_date}")

        # Base LLM
        print("[INFO] Initializing LLM...")
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )

        # 2) Retrieval (multi-query advanced)
        print("[INFO] Running retrieval agent...")
        retrieval_agent = make_retrieval_agent(llm, vector_db)
        internal_info = retrieval_agent.run(question)
        print("[INFO] Retrieval agent complete.")

        # 3) Grader
        print("[INFO] Running grader agent...")
        is_sufficient = grader_agent(llm, internal_info)
        print(f"[INFO] Is sufficient info: {is_sufficient}")

        external_info = ""
        if not is_sufficient:
            # 4a) Finance data
            print("[INFO] Running finance agent...")
            finance_agent = make_finance_agent(llm, start_date, end_date)
            finance_info = finance_agent.run(f"Get me market data for {ticker}")
            print("[INFO] Finance agent complete.")

            external_info = finance_info 

        # 6) Recommender
        print("[INFO] Running recommender agent...")
        final_answer = recommender_agent(llm, internal_info, external_info)
        print("[INFO] Recommender agent complete.")
        return final_answer

    except Exception as e:
        print(f"[ERROR] Error in multi-agent pipeline: {e}")
        raise

