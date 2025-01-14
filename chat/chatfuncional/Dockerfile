# Usa una imagen base oficial de Python 3.12
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos al contenedor
COPY requirements.txt /app/
COPY /src/tmq.pdf /app/
COPY /src/019-24/notas.pdf /app/
COPY /src/019-24/orden01924.pdf /app/
COPY /src/020-24/orden02024_1.pdf /app/

# Copia el archivo hello.py al contenedor
# COPY document.py /app
COPY chat2.py /app

# Instala las dependencias de Python
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Ejecuta el archivo hello.py
CMD ["python", "chat2.py"]