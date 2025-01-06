
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI

# Configurar la API Key de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Por favor, configura la variable de entorno OPENAI_API_KEY.")
    st.stop()

st.title("Analizador de PDFs y Resumen con OpenAI")

uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

if uploaded_file:
    # Leer el contenido del PDF
    pdf_reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() for page in pdf_reader.pages)

    if not text.strip():
        st.error("No se pudo extraer texto del PDF. Intenta con otro archivo.")
        st.stop()

    # Crear un documento para LangChain
    document = Document(page_content=text)

    # Configurar el modelo LLM de OpenAI
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    from langchain.prompts import PromptTemplate

    # Crear un prompt personalizado para generar resúmenes en español
    template = """
    Proporciona un resumen breve y preciso del siguiente texto en español:
    {text}
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])

    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)


    # Generar resumen
    with st.spinner("Generando resumen..."):
        summary = summarize_chain.run([document])

    st.subheader("Resumen del documento")
    st.write(summary)