FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

EXPOSE 8501
EXPOSE 11434

# Start Ollama + Streamlit properly (with model pull)
CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull gemma3:1b && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]