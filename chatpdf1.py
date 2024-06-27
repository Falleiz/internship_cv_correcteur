import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
        pdf_reader = PdfReader(pdf_docs)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversational_chain():
    prompt_template = """
        Vérifie si le document fourni est un curriculum vitae (CV). 
        Si c'est un CV, analyse-le et donne des remarques sur les erreurs d'orthographe et de grammaire.
        Si le contenu du CV n'est pas pertinent par rapport au secteur d'activité {sector}, donne des recommandations de rédaction pour l'améliorer.
        Ne fournis pas de mauvaises recommandations.
        Dans le cas contraire, félicite l'auteur pour la pertinence et la richesse de son CV dans le secteur {sector}.
        Juge la pertinence du CV en fonction du secteur d'activité spécifié par l'utilisateur : {sector}.
        Si le document soumis n'est pas un CV, informe l'utilisateur que le document soumis n'est pas un CV et demande de soumettre un CV.
        Context: {context}
        Question: {question}

        Answer: """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "sector"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, text_chunks, sector):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question, "sector": sector},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="centrale-internship")
    st.header("Optimisez votre CV et obtenez des conseils de rédaction personnalisés.")

    user_question = "Analyse le CV"

    st.title("Menu:")
    pdf_docs = st.file_uploader("Téléchargez vos fichiers PDF et cliquez sur le bouton Soumettre et Traiter", accept_multiple_files=True)
    sector = st.text_input("Entrez le secteur d'activité où vous souhaitez faire un stage")
    if st.button("Soumettre et Traiter"):
        with st.spinner("Processing..."):
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.success("Done")
                if user_question and text_chunks and sector:           
                    user_input(user_question, text_chunks, sector)
            else:
                st.error("Veuillez télécharger au moins un fichier PDF.")
            if not sector:
                st.error("Veuillez entrer un secteur d'activité.")

if __name__ == "__main__":
    main()

