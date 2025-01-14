from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Códigos de escape ANSI para colores
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"

# Iniciar chat
def iniciar_chat():
    # Cargar modelo
    llm = OllamaLLM(model="gemma2:2b", base_url="http://ollama:11434")

    # Lista de archivos PDF a cargar
    archivos_pdf = ["tmq.pdf", "notas.pdf", "orden01924.pdf"]
    documentos = []

    # Cargar documentos
    for archivo in archivos_pdf:
        try:
            loader = PyMuPDFLoader(archivo)
            documentos.extend(loader.load())
        except FileNotFoundError:
            print(f"El archivo {archivo} no se encuentra en el directorio. Continuando con los demás.")

    if not documentos:
        print("No se encontraron documentos para procesar.")
        exit(1)

    # PARTIR EL DOCUMENTO EN PEQUEÑOS PEDAZOS DE 2000 TOKENS, 
    # COMPARTIR CON UNA VENTANA DE 500 TOKENS ENTRE EL TROZO X Y EL TROZO Y ESTO PARA
    # QUE SE COMPARTA EL CONTEXTO
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(documentos)

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

    num_resultados = min(10, len(docs))  # Ajusta k según el número de resultados disponibles
    retriever = vectorstore.as_retriever(search_kwargs={'k': num_resultados})

    # Crear plantilla de prompt
    custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
    Si no sabes la respuesta, simplemente di "No lo se". No intentes inventar una respuesta.

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

    print("¡Bienvenid@ al chat! Escribe 'salir' para terminar.")
    while True:
        pregunta = input(f"{AZUL}Tú:{RESET} ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        # Intentar obtener respuesta del vectorstore
        respuesta = qa.invoke({"query": pregunta})
        
        # Verificar si no se encontró una respuesta en el PDF
        if not respuesta['result'] or "No lo se" in respuesta['result']:  # Si no se encuentra una respuesta válida o es ambigua
            respuesta_general = llm.invoke(pregunta)
            print(f"{VERDE}Asistente:{RESET} No encontré respuesta en el documento, pero busqué información general.\n{respuesta_general}")
        else:
            metadata = []
            for _ in respuesta['source_documents']:
                metadata.append(('page: ' + str(_.metadata['page']), _.metadata['file_path']))
            print(f"{VERDE}Asistente:{RESET} {respuesta['result']}\n{metadata}")

if __name__ == "__main__":
        iniciar_chat()

