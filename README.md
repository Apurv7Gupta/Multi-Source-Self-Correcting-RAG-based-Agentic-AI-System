![Architecture Diagram](./Architecture_diagram.png)

```bash

.
├── backend/
│   ├── app.py                # Orchestration & API
│   ├── ingestion.py          # Logic for PDFs/Markdown (Unstructured.io/PyMuPDF)
│   ├── requirements.txt
│   ├── .env
│   └── chroma_db/            # Local vector storage (persistent directory)
│
├── frontend/                 # Streaming UI React)
│   ├── src/
│       ├── components/
│       │   └── ChatWindow.js # Logic for handling SSE (StreamingResponse)
│       └── App.js
│
│
│
├── scripts/
│   └── init_db.sql           # PostgreSQL schema for entity memory/chat history
│
└── docker-compose.yml        # Orchestrates Postgres, Backend, and Frontend

```
