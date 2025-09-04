import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import subprocess
import requests
import os
import time
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Document Chatbot with Llama",
    page_icon="📄",
    layout="wide"
)

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
def check_ollama_model(model_name="llama3.2:1b"):
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
def pull_ollama_model(model_name="llama3.2:1b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            timeout=300,  # 5 minutes timeout for model download
            stream=True
        )
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process the streamed response
        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode('utf-8')
                    if '"status":"pulling' in data:
                        status_text.text(f"Download status: {data}")
                    elif '"completed":' in data and '"total":' in data:
                        # Update progress bar
                        progress = int(data.split('"completed":')[1].split(',')[0])
                        total = int(data.split('"total":')[1].split('}')[0])
                        if total > 0:
                            progress_bar.progress(progress / total)
                except:
                    continue
        
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"Failed to pull model: {e}")
        return False

# Initialize vector store and QA chain
def initialize_system():
    try:
        # Check if vector store exists
        if not os.path.exists("vector_store"):
            st.error("Vector store not found. Please run ingest.py first!")
            return False
        
        # Initialize local embeddings
        embeddings = FastEmbedEmbeddings()
        
        # Load existing vector store
        vector_store = Chroma(
            persist_directory="vector_store",
            embedding_function=embeddings
        )
        
        # Create custom prompt for Llama
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
        
        # Initialize Llama model through Ollama
        llm = Ollama(
            model="llama3.2:1b",
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

# Sidebar
with st.sidebar:
    st.title("🦙 Llama Document Chatbot")
    st.markdown("""
    **How to use:**
    1. Install Ollama from https://ollama.com/
    2. Place documents in `docs/` folder
    3. Run `python ingest.py`
    4. Initialize the system using the button below
    5. Ask questions about your documents!
    """)
    
    st.divider()
    
    # Ollama status section
    st.subheader("Ollama Status")
    
    if st.button("🔍 Check Ollama Status"):
        if check_ollama_service():
            st.success("✅ Ollama service is running!")
            
            if check_ollama_model():
                st.success("✅ Llama 3.2:1B model is available!")
            else:
                st.warning("⚠️ Llama 3.2:1B model not found.")
                if st.button("📥 Download Llama 3.2:1B Model"):
                    with st.spinner("Downloading model... This may take several minutes."):
                        if pull_ollama_model():
                            st.success("✅ Model downloaded successfully!")
                        else:
                            st.error("❌ Failed to download model.")
        else:
            st.error("❌ Ollama is not running. Please install and start Ollama.")
            st.info("Download from: https://ollama.com/")
            st.info("After installing, open a terminal and run: `ollama serve`")
    
    st.divider()
    
    # System initialization
    st.subheader("System Initialization")
    
    def initialize_system_handler():
        # Check Ollama first
        if not check_ollama_service():
            st.error("Cannot initialize: Ollama is not running.")
            return False
            
        # Check if model exists
        if not check_ollama_model():
            st.warning("Llama 3.2:1B model not found. Downloading now...")
            if not pull_ollama_model():
                st.error("Failed to download model. Cannot initialize system.")
                return False
                
        # Initialize the system
        with st.spinner("Loading documents and initializing model..."):
            if initialize_system():
                st.success("✅ System ready! You can now ask questions.")
                return True
            else:
                st.error("❌ Failed to initialize system.")
                return False
    
    if st.button("🔄 Initialize System"):
        initialize_system_handler()

# Main chat interface
st.title("🦙 Document Chatbot with Llama 3.2")
st.markdown("Ask questions about your uploaded documents - running locally with Ollama!")

# Display system status
status_col1, status_col2 = st.columns(2)
with status_col1:
    if st.session_state.ollama_status == "running":
        st.success("✅ Ollama: Running")
    elif st.session_state.ollama_status == "not_running":
        st.error("❌ Ollama: Not Running")
    else:
        st.info("🔍 Ollama: Status Unknown")
        
with status_col2:
    if st.session_state.system_ready:
        st.success("✅ System: Initialized")
    else:
        st.warning("⚠️ System: Not Initialized")

st.divider()

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
            st.error("Please initialize the system first using the sidebar buttons.")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "System not initialized. Please check Ollama status and initialize system."
        })
    else:
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Llama is thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    response = result["result"]
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show source documents
                    if "source_documents" in result and result["source_documents"]:
                        with st.expander("📁 View source documents"):
                            for i, doc in enumerate(result["source_documents"]):
                                source_name = doc.metadata.get('source', 'Unknown')
                                st.markdown(f"**Source {i+1}:** `{os.path.basename(source_name)}`")
                                st.text(doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
                                st.divider()
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

# Footer
st.divider()
st.caption("Powered by Llama 3.2:1B via Ollama • Running locally on your machine")
