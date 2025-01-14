from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

client = QdrantClient("http://localhost:6333")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

# Conecta a la colección de documentos en Qdrant
vector_store = QdrantVectorStore(
    embedding=embeddings,
    client=client,
    collection_name="my_documents"
)

# Define una función para realizar preguntas y obtener respuestas
def ask_question(question):
    # Limpia caracteres no ASCII
    question = "".join(c for c in question if ord(c) < 128)

    # Recuperar documentos relevantes
    documents = vector_store.search(question, "similarity", k=5)

    # Imprimir los documentos para verificar relevancia
    for i, doc in enumerate(documents):
        print(f"Documento {i + 1}: {doc.page_content}")

    # Resumir el contexto
    context = " ".join(doc.page_content for doc in documents[:3])

    # Formatear el input del modelo
    input_text = f"""
    Responde de forma precisa a la siguiente pregunta basándote en el contexto proporcionado.
    Pregunta: {question}
    Contexto: {context}
    """

    # Generar respuesta
    response = llm.invoke(input=input_text)
    return response



# Realiza una pregunta
question = "Dame solo el telefono de Turbomaquinas SA de CV"
response = ask_question(question)
print(response)