import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# pas la bibliothéque mais plutot le fichier google.py !!!!!!

from google.colab import drive



# VARIABLE GLOBALE de mise en cache
_CACHED_RETRIEVER = None




# 1. récupération des embeddings de chromadb et colab

def get_embeddings()-> Chroma:
    drive.mount('/content/drive')
    colab_persist_dir = Path('/content/drive/MyDrive/Chatbot_Loi_Travail/chroma_db')
    local_persist_dir = (
        Path(__file__).resolve().parents[1]
        / 'content' / 'drive' / 'Chatbot_Loi_Travail' / 'chroma_db' / 'chroma_db'
    )
    persist_dir = (
        Path(os.environ['CHROMA_PERSIST_DIRECTORY'])
        if os.environ.get('CHROMA_PERSIST_DIRECTORY')
        else local_persist_dir if (local_persist_dir / 'chroma.sqlite3').exists()
        else colab_persist_dir
    )
    
    # 2. initialisation du modèle embeddings et chargement des documents de chromadb
    embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vector_store = Chroma(
        persist_directory= persist_dir,
        embedding_function= embedding_model
    )
    return vector_store



# 2. # Avant d'appliquer bm25 retriever, on va extraire tous les articles dans chromadb sous 
    # forme  de text brut, car chromadb stocke les vecteurs mais n'a pas de moteur bm25 intégré
    
def convert_to_text(vector_store: Chroma) -> list:
    raw_data = vector_store._collection.get()
    documents = []
    for content, metadata in zip(raw_data['documents'], raw_data['metadatas']):
        documents.append(Document(page_content= content, metadata = metadata))
    if not documents:
        raise ValueError(
            "La base Chroma ne contient aucun document. Vérifiez CHROMA_PERSIST_DIRECTORY "
            "ou exécutez d'abord data_preparation.py."
        )
    return documents


# 3. Recherche hybride (RRF déjà inlcu dans EnsembleRetriever!)
def hybrid_search(documents: list, vector_store: Chroma, top_k = 10) -> EnsembleRetriever:
    bm25_retriever = BM25Retriever.from_documents(documents= documents)
    bm25_retriever.k = top_k
    
    chroma_retriever = vector_store.as_retriever(search_kwargs= {"k": top_k})
    
    # recherche hybride:
    hybrid_retriever = EnsembleRetriever(
        retrievers= [bm25_retriever, chroma_retriever],
        weights= [0.5, 0.5] # meme poid pour les deux retrievers
    )
    
    return hybrid_retriever

# 4. Reranking, on garde juste top n pertinents
def reranking(hybrid_retriever: EnsembleRetriever, top_n = 5) -> ContextualCompressionRetriever:
    
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    
    compressor = CrossEncoderReranker(
        model = model,
        top_n = top_n
    )
    
    final_retriever = ContextualCompressionRetriever(
        base_compressor= compressor,
        base_retriever= hybrid_retriever
    )
    
    return final_retriever


# 5.  Pipeline finale

def build_final_retriever(top_k: int = 10, top_n: int = 5) -> ContextualCompressionRetriever:
    global _CACHED_RETRIEVER
    if _CACHED_RETRIEVER is not None:
        return _CACHED_RETRIEVER  # Retourne l'instance déjà en mémoire (Instantané !)
        
    print("Initialisation unique du retriever hybride et du reranker...")
    
    vector_store = get_embeddings()
    documents = convert_to_text(vector_store=vector_store)
    hybrid_retriever = hybrid_search(
        documents=documents,
        vector_store=vector_store,
        top_k=top_k,
    )
    _CACHED_RETRIEVER = reranking(hybrid_retriever=hybrid_retriever, top_n=top_n)
    return _CACHED_RETRIEVER

    #return reranking(hybrid_retriever=hybrid_retriever, top_n=top_n)



"""def main():
    query = str(input("Veuillez saisir votre query: \n"))
    
    
    final_retriever = build_final_retriever(top_k= 10, top_n= 5)
    
    
    
    final_docs = final_retriever.invoke(query)
    print("Les résultats sont : \n")
    for i, doc in enumerate(final_docs, start=1):
        article = doc.metadata.get('article_no', 'N/A')
        print(f"Rang {i} [{article}] :")
        print(f"{doc.page_content[:150]}...\n")
    
    
    

if __name__ == "__main__":
    main()"""
