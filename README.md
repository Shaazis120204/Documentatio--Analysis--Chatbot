# Documentatio--Analysis--Chatbot
A locally-running document question-answering system powered by Llama 3.2 through Ollama. This Streamlit application allows you to upload documents, process them into a vector database, and ask questions about your content with source attribution—all running completely on your local machine without external APIs.

document-chatbot-llama/
├── main.py          # Streamlit application
├── ingest.py        # Document processing script
├── docs/            # Folder for your documents
├── vector_store/    # Auto-created vector database
├── requirements.txt # Python dependencies
└── README.md        # This file

Usage
Start Ollama service:

bash
ollama serve
Place your documents in the docs/ folder (PDF, TXT, DOCX supported)

Process documents:

bash
python ingest.py

Launch the chatbot:

bash
streamlit run main.py
Initialize the system through the sidebar:

Check Ollama status

Download Llama model if needed

Initialize the document system

Start asking questions about your documents!
