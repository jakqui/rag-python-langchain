from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from upload_data import cargar_documentos, crear_vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Códigos de escape ANSI para colores
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"

# Iniciar chat
def iniciar_chat(ruta_archivo):
    llm = OllamaLLM(model="llama3.2:latest", base_url="http://ollama:11434")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vectorstore = Chroma(embedding_function=embed_model,
                            persist_directory="chroma_db_dir",
                            collection_name="standford_report")
    except:
        docs = cargar_documentos(ruta_archivo)
        vectorstore = crear_vectorstore(docs)
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

    print("¡Bienvenid@ al chat! Escribe 'salir' para terminar.")
    while True:
        pregunta = input(f"{AZUL}Tú:{RESET} ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        respuesta = qa.invoke({"query": pregunta})
        print(f"{VERDE}Asistente:{RESET}", respuesta['result'])
        #metadata = []
        #for _ in respuesta['source_documents']:
            #metadata.append(('page: '+ str(_.metadata['page']), _.metada['file_path']))
            #print(f"{VERDE}Asistente:{RESET}", respuesta['result'], '\n', metadata)

if __name__ == "__main__":
        ruta_archivo = "tmq.pdf"
        iniciar_chat(ruta_archivo)

