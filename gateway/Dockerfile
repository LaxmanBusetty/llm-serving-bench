FROM python:3.11-slim
WORKDIR /app
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway/ gateway/
ENV UPSTREAM_URL=http://host.docker.internal:8000
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/metrics')" || exit 1
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "9000"]
