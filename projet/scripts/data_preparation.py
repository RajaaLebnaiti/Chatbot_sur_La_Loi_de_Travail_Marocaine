


from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import re 



# 1. Load document
def load_documents(file_path:str) -> list:
    pdf_loader_fr = PyPDFLoader(file_path= file_path)
    pages = pdf_loader_fr.load()
    return pages

# 2. Fusionner toutes les pages en un seul text
def convert_pages_to_text(pages: list)-> str:
    full_text = "\n".join([page.page_content for page in pages])
    return full_text
    
    
# 3. chunking
def chunking(full_text: str) -> list:
    pattern = r"(\nArticle\s*(?:premier|\d+))"   
    parts = re.split(pattern, full_text)
    chunks = [] 

    for i in range(1, len(parts), 2):
        title = parts[i].strip()      
        content = parts[i+1].strip()    
        
        full_article_text = f"{title}\n{content}"
        clean_title = re.sub(r"\s+", " ", title) # on remplace espace (si il existe) par " "
        
        chunks.append(Document(
            page_content=full_article_text,
            metadata={"article_no": clean_title, "source": "Code du Travail Marocain"}
        ))
    return chunks


# 4. Text cleaning
def clean_text(chunks: list) -> list:
    for chunk in chunks:
        chunk.page_content = re.sub(r"\n", " ", chunk.page_content)
        chunk.page_content = re.sub(r"\s+", " ", chunk.page_content).strip()
    return chunks



# 5. Embeddings
def get_embeddings()-> OllamaEmbeddings:
    embedding_model = OllamaEmbeddings(model= "qwen3-embedding:0.6b")
    return embedding_model

# 6. save data (embeddings) in chroma db   

def store_chromaDb(embedding_model: OllamaEmbeddings, chunks: list) -> Chroma:
    vector_store = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model
    )
    return vector_store

# Pipeline complet:

def main(file_path:str):
    pages = load_documents(file_path= file_path)
    full_text = convert_pages_to_text(pages= pages)
    chunks = chunking(full_text= full_text)
    chunks = clean_text(chunks= chunks)
    embedding_model = get_embeddings()
    vector_store = store_chromaDb(embedding_model= embedding_model, chunks= chunks)
    
    print(f"{len(chunks)} chunks indexés dans ChromaDB.")
    
    return vector_store


if __name__ == "__main__":
    main(r"C:\Users\hp\Desktop\Chatbot_sur_La_Loi_de_Travail_Marocaine\projet\data\raw\Code du travail.pdf")