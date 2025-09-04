# main.py (with multi-modal support)
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import subprocess
import requests
import os
import time
import base64
from PIL import Image as PILImage
from io import BytesIO
from ingest import ingest_documents
from PyPDF2 import PdfReader
from PIL import Image

# Set page config
st.set_page_config(
    page_title="ajs-docs-bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f3d7a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-button {
        width: 100%;
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-size: 1.1rem;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
        text-align: center;
        padding: 0.5rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        height: 70vh;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f1f1f1;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .content-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .badge-text {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .badge-table {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .badge-image {
        background-color: #fff8e1;
        color: #ff8f00;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = "unknown"

# Check if Ollama is running
def check_ollama_service():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            st.session_state.ollama_status = "running"
            return True
    except (requests.ConnectionError, requests.Timeout):
        pass
    
    # If we can't connect, try to start Ollama
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(["ollama", "serve"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        else:  # macOS/Linux
            subprocess.Popen(["ollama", "serve"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL,
                            start_new_session=True)
        
        # Wait a bit for the service to start
        time.sleep(3)
        
        # Check again
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                st.session_state.ollama_status = "running"
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
            
    except Exception as e:
        st.error(f"Failed to start Ollama: {e}")
    
    st.session_state.ollama_status = "not_running"
    return False

# Check if the required model is available
def check_ollama_model(model_name="qwen2.5vl:3b"):
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return model_name in model_names
    except:
        pass
    return False

# Pull the required model
def pull_ollama_model(model_name="qwen2.5vl:3b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            timeout=300,
            stream=True
        )
        
        # Process the streamed response
        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode('utf-8')
                except:
                    continue
        
        return True
    except Exception as e:
        st.error(f"Failed to pull model: {e}")
        return False

# Process images with vision model
def process_image_with_vision(image_path, question):
    """Process images using vision model"""
    try:
        # Encode image to base64
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Prepare prompt for vision model
        prompt = f"""
        Analyze this image and answer the question: {question}
        
        Provide a detailed description of what you see in the image and then answer the specific question.
        Be precise and focus on the visual elements.
        """
        
        # Use vision model through Ollama
        vision_llm = Ollama(
            model="qwen2.5vl:3b",
            temperature=0.1,
            base_url="http://localhost:11434"
        )
        
        # For Ollama vision models, we need to use the chat endpoint directly
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5vl:3b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_image]
                    }
                ],
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return "I couldn't analyze the image properly. Please try again."
            
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Initialize vector store and QA chain
def initialize_system():
    try:
        # Check if vector store exists
        if not os.path.exists("vector_store"):
            # If not, try to ingest documents
            if os.path.exists("docs") and any(os.listdir("docs")):
                with st.spinner("Processing documents..."):
                    ingest_documents()
            else:
                return False
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load existing vector store
        vector_store = Chroma(
            persist_directory="vector_store",
            embedding_function=embeddings
        )
        
        # Create custom prompt
        prompt_template = """You are a helpful assistant that answers questions based on the provided context.

Context information:
{context}

Question: {question}

Please provide a detailed and accurate answer based only on the context above. 
If the context doesn't contain the answer, say "I don't have enough information to answer this question."

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Initialize LLM through Ollama
        llm = Ollama(
            model="llama3.2:1b",  # Using a more reliable model
            temperature=0.1,
            num_predict=512,
            base_url="http://localhost:11434"
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        st.session_state.system_ready = True
        return True
        
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return False

# Process uploaded files
def process_uploaded_files(uploaded_files):
    docs_path = "docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(docs_path, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Ingest the documents
    try:
        ingest_documents()
        # Reinitialize the system
        initialize_system()
        return True
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return False

# Display content type badge
def content_type_badge(doc_type):
    badge_class = "badge-text"
    if doc_type == "table":
        badge_class = "badge-table"
    elif doc_type == "image":
        badge_class = "badge-image"
    
    return f'<span class="content-badge {badge_class}">{doc_type}</span>'

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>ajs-docs-bot</h2>", unsafe_allow_html=True)
    
    # New Chat button
    if st.button("💬 New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
    
    st.divider()
    
    # File upload section
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents for analysis",
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("Process Uploaded Files", use_container_width=True):
            with st.spinner("Analyzing documents..."):
                if process_uploaded_files(uploaded_files):
                    st.success("Documents analyzed successfully!")
                    time.sleep(2)
                    st.rerun()
    
    st.divider()
    
    # System status
    st.markdown("### ⚙️ System Status")
    if st.session_state.ollama_status == "running":
        st.success("✅ Ollama: Running")
    else:
        st.error("❌ Ollama: Not Running")
    
    if st.session_state.system_ready:
        st.success("✅ System: Initialized")
    else:
        st.warning("⚠️ System: Not Initialized")
    
    # Initialize button
    if st.button("🔄 Initialize System", use_container_width=True):
        with st.spinner("Setting up system..."):
            # Check Ollama first
            if not check_ollama_service():
                st.error("Ollama is not running. Please install and start Ollama.")
            else:
                # Check if model exists
                if not check_ollama_model("llama3.2:1b"):
                    st.warning("Llama 3.2:1B model not found. Downloading...")
                    if not pull_ollama_model("llama3.2:1b"):
                        st.error("Failed to download model.")
                
                # Initialize the system
                if initialize_system():
                    st.success("System ready! You can now ask questions.")
                else:
                    st.error("Failed to initialize system.")

# Main content area
st.markdown("<h1 class='main-header'>ajs-docs-bot</h1>", unsafe_allow_html=True)
st.markdown("Ask questions about your documents - now with multi-modal support for images and tables!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if system is initialized
    if not st.session_state.system_ready:
        with st.chat_message("assistant"):
            st.error("Please initialize the system first using the sidebar button.")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "System not initialized. Please check Ollama status and initialize system."
        })
    else:
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if this is about an image
                    image_keywords = ["image", "picture", "photo", "diagram", "chart", "graph"]
                    is_image_query = any(keyword in prompt.lower() for keyword in image_keywords)
                    
                    if is_image_query and st.session_state.vector_store:
                        # Try to find image documents
                        image_docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                        image_docs = [doc for doc in image_docs if doc.metadata.get('type') == 'image']
                        
                        if image_docs:
                            # Process the first image found
                            image_source = image_docs[0].metadata.get('source')
                            image_path = os.path.join("docs", image_source)
                            
                            if os.path.exists(image_path):
                                response = process_image_with_vision(image_path, prompt)
                                st.markdown(response)
                                
                                # Show the image
                                st.image(image_path, caption=f"Image: {image_source}", use_column_width=True)
                                
                                # Add to chat history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response
                                })
                            else:
                                # Fallback to regular QA
                                result = st.session_state.qa_chain({"query": prompt})
                                response = result["result"]
                                st.markdown(response)
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response
                                })
                        else:
                            # Regular text-based response
                            result = st.session_state.qa_chain({"query": prompt})
                            response = result["result"]
                            st.markdown(response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                    else:
                        # Regular text-based response
                        result = st.session_state.qa_chain({"query": prompt})
                        response = result["result"]
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                    
                    # Show source documents
                    if "source_documents" in result and result["source_documents"]:
                        with st.expander("📁 View source documents"):
                            for i, doc in enumerate(result["source_documents"]):
                                source_name = doc.metadata.get('source', 'Unknown')
                                doc_type = doc.metadata.get('type', 'text')
                                st.markdown(f"**Source {i+1}:** {content_type_badge(doc_type)} `{os.path.basename(source_name)}`")
                                st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                                st.divider()
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

# Initialize system on first load
if not st.session_state.system_ready:
    # Check if Ollama is running
    if check_ollama_service():
        # Check if model exists
        if not check_ollama_model("llama3.2:1b"):
            # Pull model silently
            pull_ollama_model("llama3.2:1b")
        
        # Initialize system
        initialize_system()
