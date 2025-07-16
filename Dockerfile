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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
