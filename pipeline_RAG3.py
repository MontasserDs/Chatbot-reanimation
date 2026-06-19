"""
Script 3 : Pipeline RAG complet
Question -> Embedding -> ChromaDB -> Groq (Mistral) -> Réponse
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── 1. Charger la clé API depuis .env ────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── 2. Chemin ChromaDB ───────────────────────────────────────
VECTORSTORE_DIR = r"C:\Users\montasser\Desktop\data_reanimation\vectorstore"

# ── 3. Charger le modèle d'embeddings (même que Script 2) ───
print(" Chargement du modèle d'embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print(" Embeddings chargés !\n")

# ── 4. Se connecter à ChromaDB déjà rempli ───────────────────
print(" Connexion à ChromaDB...")
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings
)
print(" ChromaDB connecté !\n")

# ── 5. Créer le retriever ────────────────────────────────────
# k=3 : récupérer les 3 chunks les plus pertinents
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 6. Charger le LLM Groq (Mistral) ────────────────────────
# Mistral via Groq : gratuit, rapide, supporte le français
print(" Connexion au LLM Groq...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.5  # réponses précises et cohérentes
)
print(" LLM Groq connecté !\n")

# ── 7. Prompt template ───────────────────────────────────────
prompt_template = """
Tu es un assistant médical qui aide les proches de patients en réanimation.

RÈGLES ABSOLUES :
- Ne commence JAMAIS par "Ne vous inquiétez pas"
- Ne termine JAMAIS par "Vous êtes probablement inquiet"
- AUCUNE phrase répétitive ou automatique
- Réponds STRICTEMENT dans la langue de la question :
  * Question en anglais → réponse en anglais UNIQUEMENT
  * Question en français → réponse en français UNIQUEMENT
  * Question en arabe → réponse en arabe UNIQUEMENT
  * Question en darija → réponse en darija UNIQUEMENT
- Réponds DIRECTEMENT à la question
- Maximum 3 lignes
- Pas de listes inutiles
- Ne donne jamais de conseils médicaux directs
- Si la question est trop vague, demande une précision
  au lieu d'inventer une réponse générale

Contexte : {context}

Question : {question}

Réponse :


"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ── 8. Fonction pour formater les chunks récupérés ───────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── 9. Chaîne RAG ────────────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 10. Fonction de question-réponse ─────────────────────────
def ask(question: str):
    print(f"\n{'='*50}")
    print(f" Question : {question}")
    print('='*50)
    reponse = rag_chain.invoke(question)
    print(f"\n Réponse : {reponse}")
    return reponse

# ── 11. Test du pipeline ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  PIPELINE RAG — CHATBOT RÉANIMATION")
    print("=" * 50 + "\n")

    questions = [
        "Quels sont les horaires de visite en réanimation ?",
        "Comment obtenir des informations sur l'état du patient ?",
        "Quels sont les droits des familles en réanimation ?"
    ]

    for question in questions:
        ask(question)
        print()