import os
import uuid
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient, models
import logging
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Códigos de escape ANSI para colores
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"

# Función para cargar documentos
def cargar_documentos(archivos_pdf):
    """
    Carga documentos PDF y devuelve una lista de documentos.
    """
    documentos = []
    for archivo in archivos_pdf:
        try:
            loader = PyMuPDFLoader(archivo)
            docs = loader.load()
            for i, doc in enumerate(docs):
                # Update metadata directly on each document
                doc.metadata['text'] = doc.page_content
                doc.metadata['page'] = i + 1  # Agregar el número de página
                doc.metadata['file_path'] = archivo  # Agregar el nombre del archivo
            documentos.extend(docs)
        except FileNotFoundError:
            logger.warning(f"Archivo {archivo} no encontrado")
    return documentos

# Función para inicializar el modelo de embeddings
def inicializar_modelo_embeddings():
    """
    Inicializa el modelo de embeddings.
    """
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    return modelo

# Función para crear la colección en Qdrant
def crear_coleccion_qdrant(qdrant_client, collection_name):
    """
    Crea la colección en Qdrant si no existe.
    """
    collections = qdrant_client.get_collections()
    collection_names = [col.name for col in collections.collections]
    if collection_name not in collection_names:
        logger.info("Creando nueva colección en Qdrant")
        encoder = inicializar_modelo_embeddings()
        vector_size = encoder.get_sentence_embedding_dimension()  # Esto ahora es válido
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
    else:
        logger.info("Usando colección existente en Qdrant")

# Función para subir documentos a Qdrant
def subir_documentos_a_qdrant(qdrant_client, collection_name, documentos):
    """
    Sube documentos a Qdrant.
    """
    encoder = inicializar_modelo_embeddings()
    puntos = []
    for idx, doc in enumerate(documentos):
        vector = encoder.encode(doc.metadata['text']).tolist()  # Método encode
        puntos.append(models.PointStruct(
            id=str(uuid.uuid4()),  # Genera un UUID aleatorio
            vector=vector,
            payload=doc.metadata
        ))
    qdrant_client.upsert(
        collection_name=collection_name,
        points=puntos
    )

# Función para crear el retriever de Qdrant
def crear_retriever_qdrant(qdrant_client, collection_name, documentos):
    # Use HuggingFaceEmbeddings instead of directly using SentenceTransformer
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = QdrantVectorStore.from_documents(
        documents=documentos,
        embedding=embeddings,
        url="localhost:6333",
        collection_name=collection_name
    )
    
    return vectorstore.as_retriever(search_kwargs={'k': 10})

# Función para crear la plantilla de prompt
def crear_plantilla_prompt():
    """
    Crea la plantilla de prompt.
    """
    custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
    Si no sabes la respuesta, simplemente di "No lo sé". No intentes inventar una respuesta.

    Contexto:
    {context}
    Pregunta:
    {question}

    Solo proporciona la siguiente información y responde siempre en español:
    Respuesta:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Función para crear la cadena de preguntas y respuestas
def crear_cadena_preguntas_respuestas(llm, retriever):
    """
    Crea la cadena de preguntas y respuestas.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": crear_plantilla_prompt()}
    )
    return qa




# Función principal
def iniciar_chat():
    """
    Inicia el chat.
    """
    logger.info("Iniciando sistema de RAG")
    try:
        llm = OllamaLLM(model="gemma2:2b", base_url="http://localhost:11434")
        qdrant_client = QdrantClient(host="localhost", port=6333)
    except Exception as e:
        logger.error(f"Error iniciando servicios: {e}")
        raise

    # Lista de archivos PDF a cargar
    # archivos_pdf = ["tmq.pdf", "notas.pdf", "orden01924.pdf"]
    archivos_pdf = ["tmq.pdf"]
    documentos = cargar_documentos(archivos_pdf)
    
    # Crear la colección en Qdrant
    collection_name = "my_documents"
    crear_coleccion_qdrant(qdrant_client, collection_name)
    
    # Subir documentos a Qdrant
    #subir_documentos_a_qdrant(qdrant_client, collection_name, documentos)
    
    # Crear el retriever de Qdrant
    retriever = crear_retriever_qdrant(qdrant_client, collection_name, documentos)#WARNING DEL QDRANTSTORE
    
    # Crear la plantilla de prompt
    prompt = crear_plantilla_prompt()

    # Crear la cadena de preguntas y respuestas
    qa = crear_cadena_preguntas_respuestas(llm, retriever)

    print("¡Bienvenid@ al chat! Escribe'salir' para terminar.")
    while True:
        pregunta = input(f"{AZUL}Tú:{RESET} ")
        if pregunta.lower() =='salir':
            print("¡Hasta luego!")
            break

        # Intentar obtener respuesta del vectorstore
        respuesta = qa.invoke({"query": pregunta})
        if not respuesta['result'] or "No lo sé" in respuesta['result']:
            # Si no se encuentra una respuesta válida o es ambigua
            respuesta_general = llm.invoke(pregunta)
            print(f"{VERDE}Asistente:{RESET} No encontré información en el PDF, pero busqué información general.\n{respuesta_general}")
        else:
            metadata = [f"page: {doc.metadata['page']}, {doc.metadata['file_path']}" 
            for doc in respuesta['source_documents']]
            print(f"{VERDE}Asistente:{RESET} {respuesta['result']}\n{metadata}")

if __name__ == "__main__":
    iniciar_chat()