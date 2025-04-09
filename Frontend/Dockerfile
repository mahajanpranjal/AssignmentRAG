FROM --platform=linux/amd64 python:3.9-slim-buster


# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout=30 --index-url https://mirrors.aliyun.com/pypi/simple/


# Copy Streamlit app
COPY . .

# Expose port 8501
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
