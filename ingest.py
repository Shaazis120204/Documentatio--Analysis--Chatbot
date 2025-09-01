import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

def ingest_documents():
    # Initialize embeddings using FastEmbed (more stable)
    embeddings = FastEmbedEmbeddings()
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = []
    
    # Load documents from docs folder
    docs_path = "docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print("Created docs folder. Please add documents and run again.")
        return
    
    for file in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file)
        
        try:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = file
                documents.extend(loaded_docs)
                print(f"Loaded PDF: {file}")
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = file
                documents.extend(loaded_docs)
                print(f"Loaded TXT: {file}")
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = file
                documents.extend(loaded_docs)
                print(f"Loaded DOCX: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not documents:
        print("No documents found in docs folder or failed to load documents!")
        return
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_store"
    )
    
    print("Documents ingested successfully!")
    print(f"Vector store created with {len(chunks)} chunks")

if __name__ == "__main__":
    ingest_documents()