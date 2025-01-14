from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Cargar modelo
llm = OllamaLLM(model="llama3.2:latest", base_url="http://ollama:11434")

# Cargar documentos
try:
    loader = PyMuPDFLoader("tmq.pdf")
    data_pdf = loader.load()
except FileNotFoundError:
    print("El archivo tmq.pdf no se encuentra en el directorio.")
    exit(1)

# PARTIR EL DOCUMENTO EN PEQUEÑOS PEDAZOS DE 2000 TOKENS, 
# COMPARTIR CON UNA VENTANA DE 500 TOKENS ENTRE EL TROZO X Y EL TROZO Y ESTO PARA
# QUE SE COMPARTA EL CONTEXTO
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
docs = text_splitter.split_documents(data_pdf)

# print(docs[0])

# UTILIZAR LOS EMBEDDINGS DE LOS CHUNKS CREADOS
embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cargar o crear base de datos Chroma
# TRATAR DE BUSCAR BD EXISTENTE
if os.path.exists("chroma_db_dir"):
    vectorstore = Chroma(
        persist_directory="chroma_db_dir",
        embedding_function=embed_model,
        collection_name="stanford_report_data"
    )
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_dir",
        collection_name="stanford_report_data"
    )

retriever=vectorstore.as_retriever(search_kwargs={'k':3})

# Crear plantilla de prompt
custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
Si no sabes la respuesta, simplemente di que no lo sabes. No intentes inventar una respuesta.

Contexto:
{context}
Pregunta:
{question}

Solo proporciona la siguiente información y responde siempre en español:
Respuesta:
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Crear cadena de preguntas y respuestas
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Ejecutar consulta
response = qa.invoke({"query":"¿Cúal es el teléfono y en dónde se encuentra ubicada la empresa Turbomaquinas SA de CV?"})
# Imprimir resultados
if response:
    print("Respuesta obtenida:\n", response['result'])
else:
    print("No se pudo obtener una respuesta. Revisa la configuración del modelo o del vectorstore.")