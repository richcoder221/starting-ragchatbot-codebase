#!/bin/bash

mkdir -p docs

if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your DEEPSEEK_API_KEY in .env"

cd backend && uv run uvicorn app:app --reload --port 8000
