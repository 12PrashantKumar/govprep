FROM python:3.10-slim
WORKDIR /app
COPY requirements-frontend.txt .
RUN pip install -r requirements-frontend.txt
COPY pg_app.py .

# SET DEFAULT PORT FOR CLOUD RUN
ENV PORT=8080
EXPOSE 8080

# STARTUP COMMAND (Flexible binding)
CMD streamlit run pg_app.py --server.port=$PORT --server.address=0.0.0.0