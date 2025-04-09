FROM python:3.10-slim-bullseye

# Set environment
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libmupdf-dev \
    wget \
    curl \
    git \
    pkg-config \
    make \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 🔧 Install SQLite 3.42.0 from source (Chroma requires >= 3.35.0)
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz && \
    tar xzf sqlite-autoconf-3420000.tar.gz && \
    cd sqlite-autoconf-3420000 && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && make install && \
    cd .. && rm -rf sqlite-autoconf-3420000*

# ✅ Force Python to use the updated SQLite
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PATH="/usr/local/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch & Torchvision for transformers + docling-ibm-models
RUN pip install --no-cache-dir --timeout=600 \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir sentence-transformers

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 600

# Copy code
COPY . .

# Cloud Run requires the app to listen on port 8080
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
