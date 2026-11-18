FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model weights at build time, so the container embeds offline
ARG EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV EMBEDDING_CACHE_DIR=/opt/models
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(os.environ['EMBEDDING_MODEL'], cache_dir=os.environ['EMBEDDING_CACHE_DIR'])"

# Copy application code
COPY ./app /code/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 