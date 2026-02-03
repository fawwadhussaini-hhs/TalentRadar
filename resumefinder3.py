import os
import io
import datetime
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from google.oauth2 import service_account

# LangChain & Gemini Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CONFIGURATION ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
FOLDER_ID = st.secrets["FOLDER_ID"]
DB_PATH = "./chroma_db"


# --- 2. CORE FUNCTIONS ---

def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def get_drive_service():
    # This grabs the dictionary you created in secrets.toml
    creds_info = st.secrets["gcp_service_account"]

    # Create credentials from the info dictionary
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=SCOPES
    )

    return build('drive', 'v3', credentials=creds)

def get_db_stats():
    if not os.path.exists(DB_PATH):
        return set(), 0, 0
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        data = vector_db.get(include=['metadatas', 'documents'])
        if not data or 'metadatas' not in data: return set(), 0, 0

        metas = data['metadatas']
        docs = data['documents']
        indexed_names = set()
        unreadable_names = set()

        for m, d in zip(metas, docs):
            name = m.get('source')
            if name:
                indexed_names.add(name)
                if d == "EMPTY_FILE_MARKER": unreadable_names.add(name)

        scanned_count = len(unreadable_names)
        readable_count = len(indexed_names) - scanned_count
        return indexed_names, readable_count, scanned_count
    except:
        return set(), 0, 0


def save_to_db(docs):
    if not docs: return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_PATH)


def process_batch():
    service = get_drive_service()
    existing_files, _, _ = get_db_stats()
    all_files = []
    page_token = None
    query = f"'{FOLDER_ID}' in parents and mimeType = 'application/pdf' and trashed = false"

    while True:
        results = service.files().list(q=query, pageSize=1000, pageToken=page_token,
                                       fields="nextPageToken, files(id, name)").execute()
        all_files.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if not page_token: break

    new_files = [f for f in all_files if f['name'] not in existing_files]
    if not new_files:
        st.session_state.auto_sync = False
        return

    batch_size = 5
    to_save = []
    for f in new_files[:batch_size]:
        try:
            request = service.files().get_media(fileId=f['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done: _, done = downloader.next_chunk()
            fh.seek(0)
            reader = PdfReader(fh)
            text = "".join(
                [p.extract_text().encode("utf-8", "ignore").decode("utf-8") for p in reader.pages if p.extract_text()])

            if text.strip():
                to_save.append(Document(page_content=text, metadata={"source": f['name']}))
                st.session_state.sync_logs.append(f"✅ Indexed: {f['name']}")
            else:
                to_save.append(Document(page_content="EMPTY_FILE_MARKER", metadata={"source": f['name']}))
                st.session_state.sync_logs.append(f"⚠️ Scanned: {f['name']}")
        except Exception as e:
            st.session_state.sync_logs.append(f"❌ Error: {f['name']}")
            continue

    if to_save:
        save_to_db(to_save)
        st.session_state.processed_session += len(to_save)


# --- 3. UPDATED RAG ANALYSIS WITH MEMORY ---

def get_llm_analysis(query, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # We search based on the newest query
    docs = vector_db.similarity_search(query, k=10)
    context = "\n\n".join([f"FILENAME: {d.metadata['source']}\nCONTENT: {d.page_content}"
                           for d in docs if d.page_content != "EMPTY_FILE_MARKER"])

    # Updated Prompt Template with History Placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         Role: You are an Elite Technical Recruiter and Talent Strategist with 10 years of experience in executive headhunting.
    Task: Your task is to analyze a pool of candidates from CONTEXT and identify the top matches based on the USER REQUIREMENT.
    Evaluation Framework: Initial Screening: Filter candidates based on "must-have" vs "nice-to-have" skills.
    Weighted Scoring: Score candidates out of 100 based on:
    Core Technical Skills (40%): Direct experience with required skills and requirement.
    Domain Expertise (30%): Industry-specific knowledge, technical understanding and seniority.
    Soft Skills/Culture Fit (10%): Indicators of leadership, communication, or adaptability.
    Location/Logistics (20%): Proximity or relocation feasibility.
    Thinking Process: Before presenting the table, provide a brief "Evaluation Summary" explaining your logic for the top-ranked candidates.
    Output Format: Present results in a Markdown table. 
    For the "Candidate Details" column, use a bulleted list format so Name, Email, and Phone appear on separate lines within the cell.
    "Score", "Address", "Expertise", "Reason" for justification
    Constraints: > - Rank the table in descending order (highest score first).
    Be critical; only assign a 100 if the candidate is a "Perfect Unicorn" match.

    CONTEXT: {context}
    USER REQUIREMENT: {question}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query,
        "chat_history": chat_history
    })
    return response.content, docs


# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="HHS Talent Radar", layout="wide")

# Persistent State Initialization
if "sync_logs" not in st.session_state: st.session_state.sync_logs = []
if "processed_session" not in st.session_state: st.session_state.processed_session = 0
if "auto_sync" not in st.session_state: st.session_state.auto_sync = False
if "messages" not in st.session_state: st.session_state.messages = []

st.title("HHS Talent Radar")

# --- DATA STATS ---
service = get_drive_service()
indexed_names, readable_count, scanned_count = get_db_stats()

# Accurate Drive Counting (Cached-like)
if "drive_total" not in st.session_state:
    all_drive_files = []
    pt = None
    while True:
        q = f"'{FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
        res = service.files().list(q=q, pageSize=1000, pageToken=pt, fields="nextPageToken, files(id)").execute()
        all_drive_files.extend(res.get('files', []))
        pt = res.get('nextPageToken')
        if not pt: break
    st.session_state.drive_total = len(all_drive_files)

pending = st.session_state.drive_total - len(indexed_names)

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Drive Total", st.session_state.drive_total)
m2.metric("Searchable", readable_count)
m3.metric("Scanned", scanned_count)
m4.metric("Pending", max(0, pending))

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    if not st.session_state.auto_sync:
        if st.button("🚀 Start Automatic Sync", use_container_width=True):
            st.session_state.auto_sync = True
            st.rerun()
    else:
        if st.button("🛑 Stop & Save", use_container_width=True):
            st.session_state.auto_sync = False
            st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Progress")
    for log in st.session_state.sync_logs[-5:]:
        st.caption(log)

# --- SYNC LOOP ---
if st.session_state.auto_sync and pending > 0:
    process_batch()
    st.rerun()

# --- CHAT INTERFACE ---
# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about candidates (e.g., 'Find me Maths teacher for primary classes', then 'Which of them have 3 years or more experience ?')"):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and show response
    if readable_count > 0:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing resumes..."):
                # Convert session history to LangChain format
                history_lc = []
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user":
                        history_lc.append(HumanMessage(content=m["content"]))
                    else:
                        history_lc.append(AIMessage(content=m["content"]))

                answer, sources = get_llm_analysis(prompt, history_lc)
                st.markdown(answer)

                # Show source expander just for the current answer
                with st.expander("View Latest Search Sources"):
                    for d in sources:
                        if d.page_content != "EMPTY_FILE_MARKER":
                            st.caption(f"File: {d.metadata['source']}")
                            st.text(d.page_content[:200] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("No data found. Please sync the database.")