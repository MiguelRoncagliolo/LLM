import streamlit as st
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
import os
import tempfile
import shutil

shutil.rmtree("./chroma_db", ignore_errors=True)


# Configuración de la app
st.set_page_config(page_title="Asistente Legal IA", layout="wide")
st.title("📚 Asistente Legal con LLM")

# Selección de modelo
modelo = st.selectbox("Selecciona el modelo a usar:", ["mistral", "llama3.2"])
llm = Ollama(model=modelo)

# Subir documento
st.subheader("1. Sube un documento legal")
archivo = st.file_uploader("Formatos aceptados: PDF, TXT", type=["pdf", "txt"])

if archivo:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(archivo.read())
        ruta_temporal = tmp_file.name

    if archivo.type == "application/pdf":
        loader = PyPDFLoader(ruta_temporal)
    elif archivo.type == "text/plain":
        loader = TextLoader(ruta_temporal, encoding="utf-8")
    else:
        st.error("Formato no soportado")
        st.stop()

    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documentos)

    st.success("Documento cargado y procesado ✅")

    # Crear embeddings y base de datos vectorial
    st.subheader("2. Procesando con Chroma...")
    embeddings = OllamaEmbeddings(model=modelo)
    persist_directory = f"./chroma_db_{modelo}"
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    retriever = vectordb.as_retriever()

    # Crear la cadena QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    st.success("Base de conocimiento lista 🧠")

    # Interfaz de preguntas
    st.subheader("3. Realiza una consulta al documento")
    pregunta = st.text_input("Escribe tu pregunta sobre el documento...")

    if pregunta:
        with st.spinner("Pensando... 🤔"):
            respuesta = qa(pregunta)
            st.markdown(f"**Respuesta:** {respuesta['result']}")
            with st.expander("🔍 Documentos usados como contexto"):
                for doc in respuesta["source_documents"]:
                    st.write(doc.metadata.get("source", "Documento"))
                    st.write(doc.page_content[:300] + "...")
else:
    st.info("Por favor sube un documento para comenzar.")
