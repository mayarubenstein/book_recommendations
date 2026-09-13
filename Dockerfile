FROM python:3.11-slim

ARG FRONTEND_REPOSITORY=https://github.com/mayarubenstein/book_recommendations.git
ARG FRONTEND_BRANCH=gettingUserInput

WORKDIR /app/backend

RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/backend
RUN git clone --depth 1 --branch "${FRONTEND_BRANCH}" "${FRONTEND_REPOSITORY}" /app/frontend \
    && pip install --no-cache-dir -r /app/frontend/requirements.txt

COPY start_space.sh /app/start_space.sh
RUN chmod +x /app/start_space.sh

ENV CATALOG_PATH=/data/all_books.json \
    EMBEDDING_ARTIFACT_PATH=/data/catalog_embeddings.npz \
    RUNTIME_CATALOG_PATH=/data/catalog_runtime.pkl \
    SPACE_DATA_DIR=/data \
    BOOK_RECOMMENDER_API_URL=http://127.0.0.1:8000

EXPOSE 7860
CMD ["/app/start_space.sh"]
