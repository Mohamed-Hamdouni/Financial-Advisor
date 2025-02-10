import streamlit as st # type: ignore
import os
import time  # Add this import
from pathlib import Path
from dotenv import load_dotenv
from utils.data_processor import initialize_database
from utils.agents import multi_agent_investment_pipeline
import base64
import pandas as pd # Import pandas

# Configure Streamlit page first
st.set_page_config(
    page_title="Financial Advisor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "vector_db"  # Assurez-vous que ce dossier existe
STATIC_DIR = Path(__file__).parent / "static"

def load_css():
    """Load custom CSS"""
    css_file = STATIC_DIR / "css/style.css"
    if (css_file.exists()):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def init_app():
    """Initialize app configuration and state"""
    # Remove info logs
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Vérifier et créer le dossier data si nécessaire
    if not DATA_DIR.exists():
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            return
        except Exception as e:
            st.error(f"Failed to create data directory: {e}")
            return
        
    # Vérifier les fichiers PDF
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        st.warning("No PDF files found. Please add PDF files to the data directory.")
        # Créer un fichier PDF de test si en développement
        if os.getenv("ENVIRONMENT") == "development":
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Test PDF for Financial Advisor", ln=1, align="C")
                test_pdf_path = DATA_DIR / "test_document.pdf"
                pdf.output(str(test_pdf_path))
                st.info(f"Created test PDF: {test_pdf_path}")
            except Exception as e:
                st.error(f"Failed to create test PDF: {e}")
        return
        
    # Afficher les PDF trouvés
    # st.sidebar.success(f"Found {len(pdf_files)} PDF files") # Remove this line
    
    # Créer le dossier vector_db s'il n'existe pas
    if not DB_DIR.exists():
        try:
            DB_DIR.mkdir(parents=True, exist_ok=True)
            # st.sidebar.info(f"Created database directory: {DB_DIR}")
        except Exception as e:
            st.error(f"Failed to create database directory: {e}")
            return
    
    # Initialiser la base de données
    if "db" not in st.session_state:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(step: str, value: float):
                progress_bar.progress(value)
                status_text.text(f"Initialization Progress: {step}")
            
            update_progress("Starting initialization...", 0.1)
            
            from langchain_openai import OpenAIEmbeddings
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY not found in environment variables")
                return
            
            update_progress("Creating embeddings...", 0.2)
            embeddings = OpenAIEmbeddings()
            
            update_progress("Initializing vector database...", 0.3)
            st.session_state.db = initialize_database(
                source_file=str(DATA_DIR),
                db_name="financial_db",
                embed_func=embeddings,
                progress_callback=update_progress  # Nouveau paramètre
            )
            
            update_progress("Database initialization complete!", 1.0)
            progress_bar.empty()
            status_text.empty()
            # st.sidebar.success("✅ Database initialized successfully!")
            
        except Exception as e:
            st.error(f"Failed to initialize database: {str(e)}")
            st.exception(e)
            st.session_state.db = None

def get_pdf_download_link(file_path: Path) -> str:
    """Generate a download link for PDF file"""
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    return f'data:application/pdf;base64,{b64_pdf}'

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h1>📊 Financial Advisor Pro</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Hide default logs
        st.write('<style>div.stDeployButton{display:none;}</style>', unsafe_allow_html=True)
        st.write('<style>div.StatusWidget-enter-done{display:none;}</style>', unsafe_allow_html=True)
        
        # Store current page in session state
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"
        
        # Custom navigation menu with icons
        menu_items = [
            ("📈", "Dashboard", "View market overview and metrics"),
            ("🔍", "Analysis", "Analyze investments and get recommendations"),
            ("📚", "Documents", "Browse and manage documents"),
            ("⚙️", "Settings", "Configure application settings")
        ]
        
        for icon, label, tooltip in menu_items:
            if st.sidebar.button(
                f"{icon} {label}",
                key=f"nav_{label}",
                help=tooltip,
                use_container_width=True
            ):
                st.session_state.current_page = label
        
        return st.session_state.current_page

def render_metrics():
    """Display key metrics"""
    cols = st.columns(3)
    
    # Calculate metrics based on available data
    num_documents = len(list(DATA_DIR.glob("*.pdf")))
    market_coverage = min(100, num_documents * 2)  # Example calculation
    ai_confidence = "High"  # Placeholder
    
    metrics = [
        ("Documents Analyzed", str(num_documents), "📚"),
        ("Market Coverage", f"{market_coverage}%", "🎯"),
        ("AI Confidence", ai_confidence, "🤖")
    ]
    
    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{icon} {label}</h3>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)

def render_dashboard():
    st.title("Investment Dashboard")
    
    # Métriques avec animation d'abord
    render_metrics()
    
    # Analyse des PDF et extraction des données pertinentes
    with st.container():
        st.markdown("""
        <div class="dashboard-card">
            <h3>Historical Performance Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        
        # Graphique des performances historiques
        with cols[0]:
            # Utiliser les données réelles des PDFs analysés
            companies = ["AAPL", "MSFT", "NVDA", "AMZN"]
            dates = ["Q1 2023", "Q2 2023", "Q3 2023"]
            
            st.markdown("### Quarterly Revenue Growth")
            chart_data = pd.DataFrame({
                "AAPL": [97.3, 81.8, 89.5],
                "MSFT": [52.7, 56.2, 56.5],
                "NVDA": [7.2, 13.5, 18.1],
                "AMZN": [127.4, 134.4, 143.1]
            }, index=dates)
            
            st.line_chart(chart_data, use_container_width=True)
            st.caption("Quarterly revenue in billions USD")

        # Graphique des retours sur investissement
        with cols[1]:
            st.markdown("### ROI Comparison")
            roi_data = pd.DataFrame({
                "Company": companies * 3,
                "Quarter": [q for q in dates for _ in range(4)],
                "ROI (%)": [
                    15.2, 12.8, 28.4, 8.7,  # Q1 2023
                    13.5, 14.2, 32.6, 10.1,  # Q2 2023
                    14.8, 15.1, 36.2, 11.3   # Q3 2023
                ]
            })
            
            st.bar_chart(
                data=roi_data,
                x="Quarter",
                y="ROI (%)",
                color="Company"
            )
            st.caption("Return on Investment percentage by company and quarter")

    # Dernières analyses
    with st.container():
        st.markdown("""
        <div class="dashboard-card">
            <h3>Recent Market Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.chat_history:
            last_analyses = st.session_state.chat_history[-3:]  # Dernières 3 analyses
            for q, a in last_analyses:
                with st.expander(q[:50] + "..."):
                    st.markdown(f"**Analysis:** {a}")

def render_analysis():
    st.title("Investment Analysis")
    
    with st.container():
        st.markdown("""
        <div class="analysis-header">
            <h3>AI Investment Analysis</h3>
            <p>Get personalized investment recommendations based on market data and financial reports.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Define question templates
    question_templates = [
        "What was Apple's revenue in 2023?",
        "What was Apple's stock price on 12/13/2023?",
        "Based on the latest news, what is your recommendation regarding AAPL stock?"
    ]
    
    # Load stock symbols from session state or use a default list
    stock_symbols = st.session_state.get('stock_symbols', ["AAPL", "MSFT", "GOOGL", "AMZN"])
    
    with st.form("analysis_form"):
        # Use question templates as examples in the text area
        query = st.text_area(
            "Your Question",
            value="\n".join(question_templates),
            help="Ask about specific stocks and time periods"
        )
        
        # Select stock symbol from dropdown
        ticker = st.selectbox(
            "Stock Symbol",
            options=stock_symbols,
            help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)"
        )
        
        if st.form_submit_button("Analyze", use_container_width=True):
            progress_text = "Analysis in progress..."
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Simulation des étapes d'analyse
                steps = [
                    ("Processing query...", 0.2),
                    ("Analyzing documents...", 0.4),
                    ("Retrieving market data...", 0.6),
                    ("Generating recommendations...", 0.8),
                    ("Finalizing analysis...", 1.0)
                ]
                
                for step_text, progress in steps:
                    status_text.text(f"{progress_text} ({int(progress * 100)}%)")
                    progress_bar.progress(progress)
                    time.sleep(0.5)  # Simulation du temps de traitement
                
                result = multi_agent_investment_pipeline(query, ticker, st.session_state.db)
                st.session_state.chat_history.append((query, result))
                
                progress_bar.empty()
                status_text.empty()
                st.success("Analysis Complete!")
                st.markdown(f"### Results\n{result}")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during analysis: {str(e)}")
                
    # Preserve form values after submit
    if "last_query" in st.session_state:
        query = st.session_state.last_query
    if "last_ticker" in st.session_state:
        ticker = st.session_state.last_ticker

    if st.session_state.chat_history:
        st.markdown("### Previous Analyses")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Analysis {i+1}: {q[:50]}..."):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")

def render_documents():
    st.title("Document Library")
    if not DATA_DIR.exists():
        st.warning("Data directory not found")
        return
    
    files = list(DATA_DIR.glob("*.pdf"))
    if not files:
        st.info("No documents available. Please add PDF files to the data directory.")
        return
    
    st.markdown("""
    <div class="document-list">
        <h3>Available Documents</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Group files by type (quarterly reports vs others)
    quarterly_reports = []
    other_docs = []
    
    for file in files:
        if any(q in file.name for q in ['Q1', 'Q2', 'Q3', 'Q4']):
            quarterly_reports.append(file)
        else:
            other_docs.append(file)
    
    # Display quarterly reports first
    if quarterly_reports:
        st.markdown("### Quarterly Reports")
        for file in sorted(quarterly_reports):
            pdf_data = get_pdf_download_link(file)
            st.markdown(
                f"""<div class="document-item">
                    <a href="{pdf_data}" download="{file.name}" class="document-link">
                        📊 {file.name}
                    </a>
                </div>""", 
                unsafe_allow_html=True
            )
    
    # Display other documents
    if other_docs:
        st.markdown("### Other Documents")
        for file in sorted(other_docs):
            pdf_data = get_pdf_download_link(file)
            st.markdown(
                f"""<div class="document-item">
                    <a href="{pdf_data}" download="{file.name}" class="document-link">
                        📄 {file.name}
                    </a>
                </div>""", 
                unsafe_allow_html=True
            )

def render_settings():
    st.title("Settings")
    
    # Load current settings
    settings = st.session_state.get('settings', {})
    
    with st.form("settings_form"):
        # Model Settings
        st.header("Model Configuration")
        model_name = st.selectbox(
            "Language Model",
            options=["gpt-3.5-turbo", "gpt-4"],
            help="Select the AI model to use for analysis"
        )
        
        temperature = st.slider(
            "Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=settings.get('TEMPERATURE', 0.0),
            step=0.1,
            help="Higher values make the output more creative but less focused"
        )
        
        # Analysis Settings
        st.header("Analysis Settings")
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Deep"],
            value=settings.get('ANALYSIS_DEPTH', "Standard"),
            help="Controls how thorough the analysis should be"
        )
        
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            value=settings.get('RISK_TOLERANCE', "Moderate"),
            help="Affects investment recommendations"
        )
        
        if st.form_submit_button("Save Settings"):
            new_settings = {
                'MODEL_NAME': model_name,
                'TEMPERATURE': temperature,
                'ANALYSIS_DEPTH': analysis_depth,
                'RISK_TOLERANCE': risk_tolerance
            }
            st.session_state.settings = new_settings
            st.success("Settings saved successfully!")
            st.rerun()  # Use st.rerun() instead

def main():
    try:
        load_css()
        
        init_app()
        
        if not hasattr(st.session_state, 'db') or st.session_state.db is None:
            st.error("Application not properly initialized. Please check the logs above.")
            return
        
        # Get current page from session state
        current_page = render_sidebar()
        
        # Render the selected page
        sections = {
            "Dashboard": render_dashboard,
            "Analysis": render_analysis,
            "Documents": render_documents,
            "Settings": render_settings
        }
        sections[current_page]()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()