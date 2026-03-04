![Architecture Diagram](./Architecture_diagram.png)

```bash

.
├── backend/
│   ├── app.py                # Orchestration & API
│   ├── ingestion.py          # Logic for PDFs/Markdown (document_loaders)
│   ├── requirements.txt
│   ├── .env
│   ├── db_config.py
│   ├── data/
│   │   └── PDFs, DOCX, TXT, MD
│   └── config/
│       └── config.co
│       └── config.yml
│       └── prompts.yml
│
├── frontend/                 # Streaming UI (React)
    ├── src/
        ├── components/
        │   └── ChatWindow.js # Logic for handling SSE (StreamingResponse)
        └── App.js
```
