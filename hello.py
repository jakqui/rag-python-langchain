from langchain_ollama import OllamaLLM

#llm = Ollama(model = "llama3")
# Usa el nombre del contenedor en la red como base_url
#llm = OllamaLLM(model="llama3.2:latest", base_url="http://ollama:11434")
llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

response = llm.invoke("Hola, ¿quién creó facebook?")
print(response)