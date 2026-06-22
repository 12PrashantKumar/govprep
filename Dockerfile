# 1. THE BASE IMAGE: Get a lightweight version of Linux with Python 3.11 pre-installed
FROM python:3.11-slim

# 2. THE WORKING DIRECTORY: Create a folder called /app inside the container and move into it
WORKDIR /app

# 3. COPY REQUIREMENTS FIRST: By copying just this file first, 
# Docker caches the pip install step, so rebuilding is faster later.
COPY requirements.txt .

# 4. INSTALL DEPENDENCIES: Run pip install inside the Linux container
RUN pip install --no-cache-dir -r requirements.txt

# DOWNLOAD THE AI MODEL: Pre-download the model so it's baked into the image
# This prevents the container from having to download 90MB every time it wakes up.

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 5. COPY THE REST OF YOUR CODE: Now copy main.py, app.py, and everything else into /app
COPY . .

# 6. SET THE PORT: Google Cloud Run will automatically provide a port, but we default to 8080 locally
ENV PORT=8080

# 7. THE STARTUP COMMAND: When the container turns on, run the FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port $PORT