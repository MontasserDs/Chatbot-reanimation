"""
app.py : Interface Streamlit du Chatbot Réanimation
"""

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── 1. Charger la clé API ────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── 2. Chemin ChromaDB ───────────────────────────────────────
VECTORSTORE_DIR = r"C:\Users\montasser\Desktop\data_reanimation\vectorstore"

# ── 3. Configuration de la page Streamlit ───────────────────
st.set_page_config(
    page_title="Chatbot Réanimation",
    page_icon="🏥",
    layout="centered"
)

# ── 4. Titre et description ──────────────────────────────────
st.title("🏥 Chatbot Réanimation")
st.markdown("""
> **Assistant pour les proches de patients en réanimation**  
> Posez vos questions sur les visites, les droits, les soins et le fonctionnement du service.
""")

# Signature
st.markdown("""
<div style='text-align: right; color: grey; font-size: 13px;'>
    Développé par <b>Montasser_Hannour</b> 
</div>
""", unsafe_allow_html=True)
st.divider()

# ── 5. Charger le pipeline RAG (mis en cache pour performance) 
@st.cache_resource(show_spinner=False, max_entries=1)
def load_rag_chain():
    # Charger embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Connecter ChromaDB
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.1
    )

    # Prompt
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

Contexte : {context}

Question : {question}

Réponse :

    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chaîne RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ── 6. Initialiser l'historique des messages ─────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 7. Afficher l'historique ─────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── 8. Zone de saisie de la question ────────────────────────
question = st.chat_input("Posez votre question ici...")

if question:
    # Afficher la question de l'utilisateur
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("⏳ Recherche en cours..."):
            rag_chain = load_rag_chain()
            reponse = rag_chain.invoke(question)
        st.markdown(reponse)

    st.session_state.messages.append({"role": "assistant", "content": reponse})

# ── 9. Sidebar avec infos ────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    Ce chatbot est basé sur des documents officiels :
    - 📄 Documents **SRLF**
    - 🏥 Livrets d'accueil **CHU**
    - ⚖️ Chartes des droits patients
    
    ---
    ⚠️ **Avertissement**  
    Cet assistant ne remplace pas l'avis médical.  
    Pour toute urgence, contactez directement le service.
    """)

    st.divider()

    # Bouton pour effacer l'historique
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()