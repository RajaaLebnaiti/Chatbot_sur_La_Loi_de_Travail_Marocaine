import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from google.colab import drive 





# 1. récupération des embeddings de chromadb et colab


def get_embeddings()-> Chroma:
    drive.mount('/content/drive')
    persist_dir = '/content/drive/MyDrive/Chatbot_Loi_Travail/chroma_db'
    
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

    
def main():
    query = str(input("Veuillez saisir votre query: \n"))
    
    vector_store = get_embeddings()
    documents = convert_to_text(vector_store= vector_store)
    hybrid_retriever = hybrid_search(documents= documents, vector_store= vector_store, top_k= 10)
    final_retriever = reranking(hybrid_retriever= hybrid_retriever, top_n= 3)
    
    final_docs = final_retriever.invoke(query)
    print("Les résultats sont : \n")
    for i, doc in enumerate(final_docs, start=1):
        article = doc.metadata.get('article_no', 'N/A')
        print(f"Rang {i} [{article}] :")
        print(f"{doc.page_content[:150]}...\n")
    
    
    

if __name__ == "__main__":
    main()
