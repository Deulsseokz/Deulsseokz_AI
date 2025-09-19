# base layer with dependencies cached
FROM python:3.11-slim as base
WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    git pkg-config default-libmysqlclient-dev build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# copy project
FROM base
COPY . /app
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
CMD ["gunicorn", "main:app", "--workers", "2", "--timeout", "90", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8001", "--limit-request-body", "52428800"]
