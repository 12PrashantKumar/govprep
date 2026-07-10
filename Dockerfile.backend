# 1. THE BASE IMAGE: Lightweight version of Linux with Python 3.11 pre-installed
FROM python:3.11-slim

# 2. THE WORKING DIRECTORY: Move into /app inside the container
WORKDIR /app

# 3. COPY REQUIREMENTS FIRST: Caches the pip install step for faster subsequent builds
COPY requirements.txt .

# 4. INSTALL DEPENDENCIES: Install your packages inside the Linux environment
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPY THE REST OF YOUR CODE: Bring api.py, scripts/, and everything else over
COPY . .

# 6. SET THE PORT: Expose port 8080 (Cloud Run's default targeting port)
EXPOSE 8080

# 7. THE STARTUP COMMAND: Pointed explicitly to your updated api.py file
CMD ["sh", "-c", "uvicorn pg_api:app --host 0.0.0.0 --port ${PORT:-8080}"]