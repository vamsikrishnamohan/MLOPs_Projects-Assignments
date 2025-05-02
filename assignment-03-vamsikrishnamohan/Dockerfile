FROM python:3.9-slim

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and model
COPY server_api.py .
COPY model.pkl .
COPY utils.py .
COPY dense_neural_class.py .

# Environment variable for model path
ENV MODEL_PATH=/app

# Expose the port
EXPOSE 8000

# Run the API
CMD ["python", "server_api.py"]