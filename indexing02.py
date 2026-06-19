# ============================================================
# Script 2 : Indexing — Embeddings + Stockage ChromaDB
# Transformer les chunks en vecteurs et les stocker
# ============================================================

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── 1. Chemin vers les PDFs ──────────────────────────────────
PDF_DIR = Path(r"C:\Users\montasser\Desktop\data_reanimation\raw_data\pdfs")

# ── 2. Dossier où ChromaDB va stocker les vecteurs ──────────
VECTORSTORE_DIR = r"C:\Users\montasser\Desktop\data_reanimation\vectorstore"

# ── 3. Charger les PDFs ──────────────────────────────────────
def load_pdfs(pdf_dir):
    documents = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f" Chargement : {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)
    print(f" Total pages chargées : {len(documents)}\n")
    return documents

# ── 4. Découper en chunks ────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f" Total chunks créés : {len(chunks)}\n")
    return chunks

# ── 5. Créer les embeddings (modèle multilingue français) ────
def create_embeddings():
    print(" Chargement du modèle d'embeddings multilingue...")
    embeddings = HuggingFaceEmbeddings(
        # Modèle multilingue qui supporte le français
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print(" Modèle d'embeddings chargé !\n")
    return embeddings

# ── 6. Stocker dans ChromaDB ─────────────────────────────────
def store_in_chromadb(chunks, embeddings):
    print(" Stockage des vecteurs dans ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR  # sauvegarde locale
    )
    print(f" {len(chunks)} chunks stockés dans ChromaDB !\n")
    return vectorstore

# ── 7. Test de recherche ─────────────────────────────────────
def test_search(vectorstore):
    print(" Test de recherche :")
    query = "Quels sont les horaires de visite en réanimation ?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"Question : {query}\n")
    for i, doc in enumerate(results):
        print(f"--- Résultat {i+1} ---")
        print(f"Source : {doc.metadata.get('source', 'inconnue')}")
        print(f"Texte  : {doc.page_content[:200]}...\n")

# ── 8. Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  INDEXING — EMBEDDINGS + CHROMADB")
    print("=" * 50 + "\n")

    documents = load_pdfs(PDF_DIR)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = store_in_chromadb(chunks, embeddings)
    test_search(vectorstore)

    print("Indexing terminé ! Vectorstore prêt.")