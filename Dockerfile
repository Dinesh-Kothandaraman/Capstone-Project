# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose necessary ports (FastAPI - 8000, Streamlit - 8501)
EXPOSE 8000 8501

# Run FastAPI and Streamlit in parallel
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
