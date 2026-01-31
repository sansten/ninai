# OCR sidecar: provides HTTP API around tesseract without requiring apt-get.
# Base image includes tesseract + python.
FROM sleeck/tesseract:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8000

CMD ["python3", "app.py"]
