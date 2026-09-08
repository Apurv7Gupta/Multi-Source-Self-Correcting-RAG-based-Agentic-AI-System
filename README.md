# This is a self-correcting RAG based chatbot

A production-grade, cloud-native RAG (Retrieval-Augmented Generation) chatbot with streaming responses, conversation memory, safety guardrails, and hybrid search (vector DB + live web).

![Architecture Diagram](./Architecture_diagram.png)

---

## Stack

| Layer         | Technology                                                       |
| ------------- | ---------------------------------------------------------------- |
| LLM           | `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API |
| Orchestration | LangGraph (StateGraph)                                           |
| Vector DB     | Pinecone                                                         |
| Web Search    | Tavily Search API                                                |
| Guardrails    | NVIDIA NeMo Guardrails                                           |
| Memory        | PostgreSQL via Neon                                              |
| API           | FastAPI with SSE streaming                                       |
| Frontend      | React                                                            |
| Deployment    | Render (backend) + Vercel (frontend)                             |

---

## Project Structure

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

---

## Environment Variables

Create a `.env` file in `backend/`:

```env
# HuggingFace
HF_TOKEN=

# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=collection1

# Tavily
TAVILY_API_KEY=

# Neon (PostgreSQL)
PGSQL_USERNAME=
PGSQL_PASSWORD=
PGSQL_HOST=
PGSQL_PORT=5432
PGSQL_NAME=neondb
```

---

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Ingest documents

Place your PDFs, DOCX, or Markdown files in `backend/data/`, then run:

```bash
python ingestion.py
```

You can also ingest a web URL programmatically:

```python
from ingestion import ingest_web_url
ingest_web_url("https://example.com/page")
```

### 3. Run locally

```bash
uvicorn app:api --reload
```

---

## API

### `POST /chat`

Streams the chatbot response as Server-Sent Events (SSE).

**Query params:**

| Param       | Type  | Description                                 |
| ----------- | ----- | ------------------------------------------- |
| `user_id`   | `str` | Identifier for the user                     |
| `thread_id` | `str` | Conversation thread (persisted in Postgres) |
| `message`   | `str` | User's message                              |

**Response stream format:**

```
data: [STATUS] Scanning PDFs and searching the web...
data: [STATUS] Finalizing response...
data: <final AI response>
```

---

## How It Works

1. **User sends a message** → FastAPI receives it
2. **Retrieve node** — runs in parallel:
   - Semantic search over ingested documents (Pinecone)
   - Live web search (Tavily)
3. **LLM node** — prompt is constructed with retrieved context + conversation history, passed through NeMo Guardrails, then answered by Llama 3.1
4. **Response streamed** back to frontend via SSE
5. **Conversation saved** to Neon PostgreSQL via LangGraph's `AsyncPostgresSaver`

---

## Guardrails

NeMo Guardrails enforce three checks on every response:

- **Input check** — blocks harmful or policy-violating user messages
- **Output check** — ensures the model response is appropriate
- **Fact check** — verifies the response is grounded in retrieved context

Configuration lives in `backend/config/`.

## External Services (All Free Tier)

| Service     | Purpose                    | Link                                     |
| ----------- | -------------------------- | ---------------------------------------- |
| HuggingFace | LLM + Embeddings inference | [huggingface.co](https://huggingface.co) |
| Pinecone    | Vector database            | [pinecone.io](https://pinecone.io)       |
| Tavily      | Web search API             | [tavily.com](https://tavily.com)         |
| Neon        | Serverless PostgreSQL      | [neon.tech](https://neon.tech)           |
| Render      | Backend hosting            | [render.com](https://render.com)         |
| Vercel      | Frontend hosting           | [vercel.com](https://vercel.com)         |
