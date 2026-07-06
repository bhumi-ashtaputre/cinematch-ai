import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# 1. Force a clean full-screen canvas layout
st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")

# 2. Premium Streaming Platform UI Presentation Matrix (CSS Injector)
st.markdown("""
    <style>
    /* Global Canvas Styling Reset */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], 
    [data-testid="stApp"], .main, [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
        background-color: #0d0f12 !important;
        color: #ffffff !important;
    }
    
    /* Remove native container padding overheads */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 7rem !important;
        max-width: 1100px !important;
        margin: 0 auto !important;
    }

    /* Top Bar Premium Mock Navigation Panel styling */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 0px;
        border-bottom: 1px solid #1a1e24;
        margin-bottom: 45px;
        width: 100%;
    }
    .nav-logo {
        color: #E50914;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: bold;
        font-size: 1.7rem;
        letter-spacing: 0.5px;
    }
    .nav-links {
        display: flex;
        gap: 35px;
        color: #9ca3af;
        font-size: 1rem;
    }
    .nav-links span.active {
        color: #ffffff;
        border-bottom: 2px solid #E50914;
        padding-bottom: 6px;
        font-weight: 600;
    }

    /* Premium Banner Display Headers */
    .brand-hero {
        text-align: center;
        margin-bottom: 35px;
    }
    .main-title {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 800;
        font-size: 3.4rem;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #9ca3af;
        font-style: italic;
        font-size: 1.3rem;
        margin-bottom: 25px;
    }
    .search-caption {
        background-color: #14171d;
        border: 1px solid #222730;
        border-radius: 12px;
        padding: 20px 35px;
        color: #9ca3af;
        font-size: 1.15rem;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Chat Conversation Bubble Interface styling */
    .chat-container {
        max-width: 850px;
        margin: 30px auto;
    }
    .user-bubble {
        background-color: #1c212e;
        border: 1px solid #2b3347;
        border-radius: 20px 20px 4px 20px;
        padding: 14px 22px;
        color: #ffffff;
        font-size: 1.05rem;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 25px;
        text-align: right;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .assistant-response-wrapper {
        background-color: #13161c;
        border: 1px solid #202530;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 35px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .analysis-header {
        color: #E50914;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }
    .pitch-text {
        color: #e5e7eb;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* CRITICAL INPUT FIX: White background box with solid black text entries */
    [data-testid="stChatInput"],
    .stChatInputContainer {
        background-color: #ffffff !important;
        border: 1px solid #ced4da !important;
        border-radius: 24px !important;
    }
    [data-testid="stChatInput"] textarea,
    .stChatInputContainer textarea,
    .stChatInputContainer textarea:focus,
    .stChatInputContainer textarea:active,
    .stChatInputContainer [data-testid="stWidgetLabel"] p {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    .stChatInputContainer textarea::placeholder {
        color: #71717a !important;
        -webkit-text-fill-color: #71717a !important;
    }
    footer, [data-testid="stBottom"] > div {
        background-color: #0d0f12 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Render Top Mock Navigation Panel
st.markdown("""
    <div class="top-nav">
        <div class="nav-logo">CineMatch AI</div>
        <div class="nav-links">
            <span class="active">Discover</span>
            <span>Library</span>
            <span>Trends</span>
        </div>
        <div style="color: #9ca3af; font-size: 1.2rem; cursor: pointer;">⚙️</div>
    </div>
""", unsafe_allow_html=True)

# 4. Main Banner Display Layout
st.markdown("""
    <div class="brand-hero">
        <h1 class="main-title">🎬 CineMatch AI</h1>
        <p class="subtitle">Your Intelligent Cinematic Concierge</p>
        <div class="search-caption">
            Search our vector catalog by mood, abstract concepts, or specific cinematic vibes. <br>
            <strong>Just type what you're feeling below.</strong>
        </div>
    </div>
""", unsafe_allow_html=True)

# 5. Security Guard Verification — works on both local (.env) and Streamlit Cloud (Secrets)
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ 'GEMINI_API_KEY' not found! Set it in your environment or in Streamlit Secrets.")
    st.stop()
os.environ["GEMINI_API_KEY"] = api_key

# 6. Production Ingestion Engine Layer
@st.cache_resource
def load_and_index_db():
    csv_path = "movies.csv"
    if not os.path.exists(csv_path):
        st.error(f"❌ Core data file '{csv_path}' not found!")
        st.stop()
        
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        page_content = f"Title: {row['title']}. Genre: {row['genre']}. Plot: {row['plot']}"
        doc = Document(page_content=page_content, metadata={"title": row["title"]})
        documents.append(doc)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    vector_db = Chroma.from_documents(documents, embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 2})

# Initialize components
try:
    retriever = load_and_index_db()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
except Exception as e:
    st.error(f"🔴 Initialization failed: {type(e).__name__}: {e}")
    st.stop()

# 7. Session State Chat History Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Viewport wrapper container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Re-render chat ledger entries beautifully
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="assistant-response-wrapper">
                <div class="analysis-header">🚨 CINEMATCH ANALYSIS</div>
                <div class="pitch-text">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 8. Interactive Input Box with Custom Prompt Engineering
if user_input := st.chat_input("What kind of movie vibe are you looking for today?"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Processing vector matches..."):
        try:
            # 1. Fetch the relevant source context documents from ChromaDB
            matched_docs = retriever.invoke(user_input)
            
            # 2. Build a customized system prompt structure to guide Gemini's response formatting
            context_text = "\n\n".join([doc.page_content for doc in matched_docs])
            system_prompt = f"""
            You are CineMatch AI, an elite cinematic concierge. The user is looking for a specific movie vibe: "{user_input}"
            
            Based ONLY on the following movie catalog data, provide a list of relevant recommendations:
            {context_text}
            
            For each recommended movie, format your response beautifully:
            - Write the Movie Title in bold.
            - Provide a brief, compelling 2-sentence description explaining EXACTLY why the user should watch it based on their mood/vibe request.
            
            Be engaging, cinematic, and professional. If no movies match the vibe, politely suggest trying another mood phrase.
            """
            
            # 3. Direct invoke to get a high-fidelity formatted pitch
            response = llm.invoke(system_prompt)
            ai_pitch = response.content
            
            # Save assistant payload to logs
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_pitch
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Execution Error: {e}")