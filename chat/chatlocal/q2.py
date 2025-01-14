import os
import requests
import uuid
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient, models
import logging
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Códigos de escape ANSI para colores
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"

class OllamaEmbeddings(Embeddings):
    def __init__(self, model="nomic-embed-text:latest", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _embed(self, texts):
        """
        Internal method to generate embeddings using the Ollama API
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text,
                "stream": False
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings", 
                    json=payload
                )
                response.raise_for_status()
                embedding = response.json()['embedding']
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Create a 768-dimensional embedding with zeros
                embeddings.append([0] * 768)
        
        return embeddings

    def embed_documents(self, texts):
        """
        Embed multiple documents.
        """
        return self._embed(texts)

    def embed_query(self, text):
        """
        Embed a single query.
        """
        return self._embed([text])[0]

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
                doc.metadata['text'] = doc.page_content
                doc.metadata['page'] = i + 1
                doc.metadata['file_path'] = archivo
            documentos.extend(docs)
        except FileNotFoundError:
            logger.warning(f"Archivo {archivo} no encontrado")
    return documentos

# Función para inicializar el modelo de embeddings
def inicializar_modelo_embeddings():
    """
    Inicializa el modelo de embeddings de Ollama.
    """
    modelo = OllamaLLM(model="nomic-embed-text:latest", base_url="http://localhost:11434")
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
        # Usar un tamaño de vector conocido para nomic-embed-text
        vector_size = 768  # Tamaño típico para este modelo
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
        vector = encoder.encode(doc.metadata['text']).tolist()
        puntos.append(models.PointStruct(
            id=str(uuid.uuid4()),  # Genera un UUID aleatorio
            vector=vector,
            payload=doc.metadata
        ))
    qdrant_client.upsert(
        collection_name=collection_name,
        points=puntos
    )

def eliminar_coleccion_qdrant(qdrant_client, collection_name):
    """
    Elimina una colección de Qdrant.
    """
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        logger.info(f"Colección '{collection_name}' eliminada con éxito.")
    except Exception as e:
        logger.error(f"Error al eliminar la colección '{collection_name}': {e}")

# Función para crear el retriever de Qdrant
def crear_retriever_qdrant(qdrant_client, collection_name, documentos):
    # Use OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    texts = [doc.page_content for doc in documentos]
    metadatas = [doc.metadata for doc in documentos]
    
    vectorstore = QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url="localhost:6333",
        collection_name=collection_name
    )
    
    return vectorstore.as_retriever(search_kwargs={'k': 5})

# Función para crear la plantilla de prompt
# Crear plantilla de prompt
def crear_plantilla_prompt():
    custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
    Si no sabes la respuesta, simplemente di "No lo sé". No intentes inventar una respuesta.

    Contexto:
    {context}
    Pregunta:
    {question}

    Responde SIEMPRE en ESPAÑOL, nunca en otro idioma:
    Respuesta:
    """
#Respuesta (máximo 200 caracteres, incluye referencias si es posible):

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

# Función para resumir documentos largos
def resumir_documentos(documentos):
    """
    Resumir el contenido de los documentos si son muy largos.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    documentos_resumidos = []
    for doc in documentos:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            # Crear instancias de Document para cada fragmento de texto
            documento_resumido = Document(
                page_content=chunk, 
                metadata=doc.metadata  # Mantener la metadata original
            )
            documentos_resumidos.append(documento_resumido)
    return documentos_resumidos

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
    archivos_pdf = ["tmq.pdf", "notas.pdf", "orden01924.pdf", "orden02024_1.pdf"]
    documentos = cargar_documentos(archivos_pdf)
    
    # Resumir los documentos si son grandes
    documentos_resumidos = resumir_documentos(documentos)

    collection_name = "my_documents"

    # Eliminar la colección antes
    eliminar_coleccion_qdrant(qdrant_client, collection_name)
    
    # Crear la colección en Qdrant    
    crear_coleccion_qdrant(qdrant_client, collection_name)
    
    # Subir documentos a Qdrant
    #subir_documentos_a_qdrant(qdrant_client, collection_name, documentos_resumidos)
    
    # Crear el retriever de Qdrant
    retriever = crear_retriever_qdrant(qdrant_client, collection_name, documentos_resumidos)
    
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
        
        # Mostrar los documentos de origen
        logger.info(f"Documentos recuperados: {[doc.metadata['file_path'] for doc in respuesta['source_documents']]}")
        
        if not respuesta['result'] or "No lo sé" in respuesta['result']:
            respuesta_general = llm.invoke(f"Por favor, responde a la siguiente pregunta en español: {pregunta}")
            print(f"{VERDE}Asistente:{RESET} No encontré información en el PDF, pero busqué información general.\n{respuesta_general}")
        else:
            metadata = [f"page: {doc.metadata['page']}, {doc.metadata['file_path']}" 
                        for doc in respuesta['source_documents']]
            print(f"{VERDE}Asistente:{RESET} {respuesta['result']}")#\n{metadata}

if __name__ == "__main__":
    iniciar_chat()
