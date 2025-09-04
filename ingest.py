# ingest.py (with multi-modal support)
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
import pytesseract
from PIL import Image
import tempfile

def extract_text_from_image(image_path):
    """Extract text from images using OCR"""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text if text.strip() else "No text found in image"
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Could not extract text from image"

def process_pdf_with_multimodal_content(pdf_path, filename):
    """Process PDFs containing text, images, and tables"""
    try:
        # Extract elements from PDF using unstructured
        elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=800,
        )
        
        documents = []
        for i, element in enumerate(elements):
            content = ""
            content_type = "text"
            
            if hasattr(element, 'text') and element.text.strip():
                content = element.text
                if hasattr(element, 'category'):
                    content_type = element.category.lower()
            
            elif hasattr(element, 'metadata') and element.metadata.get('text_as_html'):
                content = f"Table: {element.metadata['text_as_html']}"
                content_type = "table"
            
            if content:
                doc = {
                    "page_content": content,
                    "metadata": {
                        "source": filename,
                        "type": content_type,
                        "page_number": getattr(element, 'metadata', {}).get('page_number', 1),
                        "element_id": i
                    }
                }
                documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        # Fallback to regular PDF loader
        return []

def ingest_documents():
    # Use HuggingFace embeddings as fallback (more reliable than FastEmbed)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = []
    docs_path = "docs"
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print("Created docs folder. Please add documents and run again.")
        return
    
    for file in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file)
        
        try:
            if file.endswith('.pdf'):
                # Try advanced processing first
                pdf_docs = process_pdf_with_multimodal_content(file_path, file)
                if pdf_docs:
                    documents.extend(pdf_docs)
                    print(f"Processed PDF with multi-modal content: {file}")
                else:
                    # Fallback to regular PDF loader
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata['source'] = file
                        doc.metadata['type'] = 'text'
                    documents.extend(loaded_docs)
                    print(f"Loaded PDF: {file}")
                    
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = file
                    doc.metadata['type'] = 'text'
                documents.extend(loaded_docs)
                print(f"Loaded TXT: {file}")
                
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = file
                    doc.metadata['type'] = 'text'
                documents.extend(loaded_docs)
                print(f"Loaded DOCX: {file}")
                
            elif file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                # Process images with OCR
                image_text = extract_text_from_image(file_path)
                doc = {
                    "page_content": f"Image content: {image_text}",
                    "metadata": {
                        "source": file,
                        "type": "image"
                    }
                }
                documents.append(doc)
                print(f"Processed image with OCR: {file}")
                
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not documents:
        print("No documents found in docs folder or failed to load documents!")
        return
    
    # Convert to Document objects if needed
    from langchain.schema import Document
    final_docs = []
    for doc in documents:
        if isinstance(doc, dict):
            final_docs.append(Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            ))
        else:
            final_docs.append(doc)
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(final_docs)
    print(f"Split into {len(chunks)} chunks")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_store"
    )
    
    print("Documents ingested successfully with multi-modal support!")
    print(f"Vector store created with {len(chunks)} chunks")

if __name__ == "__main__":
    ingest_documents()
