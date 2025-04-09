FROM --platform=linux/amd64 python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Copy the DAG file
COPY dags/etl_pipeline_dag.py /app/dags/etl_pipeline_dag.py

# Set environment variables
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__WORKERS=1

# Expose port 8080
EXPOSE 8080

# Set the entrypoint to run the Airflow webserver
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "airflow.www.app:create_app()"]
