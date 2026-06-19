
# Script 1 : Preprocessing des PDFs
# Lecture + Nettoyage + Découpage en chunks

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Chemin vers les PDFs ──────────────────────────────────
PDF_DIR = Path(r"C:\Users\montasser\Desktop\data_reanimation\raw_data\pdfs")

# ── 2. Charger tous les PDFs ─────────────────────────────────
def load_pdfs(pdf_dir):
    documents = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f" Chargement : {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)
    print(f"\n Total pages chargées : {len(documents)}")
    return documents

# ── 3. Découper en chunks 
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # taille de chaque chunk en caractères
        chunk_overlap=50,     # chevauchement entre chunks
        separators=["\n\n", "\n", ".", " "]  # points de découpe
    )
    chunks = splitter.split_documents(documents)
    print(f" Total chunks créés : {len(chunks)}")
    return chunks

# ── 4. Afficher un aperçu 
def preview_chunks(chunks, n=3):
    print(f"\n Aperçu des {n} premiers chunks :\n")
    for i, chunk in enumerate(chunks[:n]):
        print(f"--- Chunk {i+1} ---")
        print(f"Source : {chunk.metadata.get('source', 'inconnue')}")
        print(f"Page   : {chunk.metadata.get('page', '?')}")
        print(f"Texte  : {chunk.page_content[:200]}...")
        print()

# ── 5. Main       
if __name__ == "__main__":
    print("=" * 50)
    print("  PREPROCESSING DU CORPUS RÉANIMATION")
    print("=" * 50 + "\n")

    # Charger les PDFs
    documents = load_pdfs(PDF_DIR)

    # Découper en chunks
    chunks = split_documents(documents)

    # Aperçu
    preview_chunks(chunks)

    print(" Preprocessing terminé !")