import os
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from nemoguardrails import LLMRails, RailsConfig
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from db_config import get_vector_db
from fastapi.middleware.cors import CORSMiddleware


# --- 1. LLM CONFIGURATION ---
llm_tool_error = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.1,
    do_sample=True,
    max_new_tokens=512,
    streaming=True,
)

llm = ChatHuggingFace(llm=llm_tool_error)
# Global variables for the lifespan
retriever = None
graph_app = None 
pool = None
# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    status: str
    temperature: float
    max_new_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    system_prompt: str
    web_search_enabled: bool
    rag_enabled: bool
# --- 3. NODES / AGENT LOGIC ---


# 1). INITIALIZE GUARDRAILS ---
config = RailsConfig.from_path("./config")
rails = LLMRails(config, llm=llm)  # Pass existing LLM


# 1.5) SEARCH THE WEB (tool) ---

# search_tool = TavilySearchResults(k=3)

search_tool = TavilySearch(max_results=3, search_depth="basic")


@tool
async def web_search(query: str):
    """
    Search the web for real-time information or topics not found in the internal Docs.
    Use this only when the user asks about current events or specific external data, or something you really don't know
    """
    results = await search_tool.ainvoke(query)

     # case 1: structured dict
    if isinstance(results, dict):
        text = ""
        if results.get("answer"):
            text += str(results["answer"]) + "\n\n"

        if "results" in results:
            text += "\n".join(
                r.get("content", "") for r in results["results"]
                if isinstance(r, dict)
            )

    # case 2: list of results
    elif isinstance(results, list):
        text = "\n".join(
            r.get("content", str(r)) if isinstance(r, dict) else str(r)
            for r in results
        )

    # case 3: plain string
    else:
        text = str(results)

    return f"--------- WEB SEARCH RESULTS ---------\n\n{text}"


# 2). RETRIEVE TOOL ---


@tool
async def retrieve_docs(query: str):
    """Fetch relevant internal documents to answer user queries about the company, guidelines, or products."""
    docs = await retriever.ainvoke(query) if retriever else []
    joined_results = "\n".join(doc.page_content for doc in docs)
    return f"--- INTERNAL DOCS KNOWLEDGE ---\n{joined_results}\n\n"


#  3). CALL MODEL NODE WITH GUARDRAILS & PROMPT TEMPLATE ---
async def call_model_node(state: AgentState):
    temp = state.get("temperature", 0.1)
    if temp <= 0:
        temp = 0.01

    max_tokens = state.get("max_new_tokens", 512)
    top_p = state.get("top_p", 0.9)
    top_k = state.get("top_k", 50)
    rep_penalty = state.get("repetition_penalty", 1.0)
    sys_prompt = state.get("system_prompt", "")
    use_web = state.get("web_search_enabled", True)
    use_rag = state.get("rag_enabled", True)

    print(f"--- DEBUG: LLM Temperature set to {temp} ---")

    llm_tool_dynamic = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        temperature=temp,
        do_sample=True,
        max_new_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep_penalty,
        streaming=True,
    )
    llm_dynamic = ChatHuggingFace(llm=llm_tool_dynamic)
    
    active_tools = []
    if use_web:
        active_tools.append(web_search)
    if use_rag:
        active_tools.append(retrieve_docs)

    if active_tools:
        llm_with_tools_dynamic = llm_dynamic.bind_tools(active_tools)
    else:
        llm_with_tools_dynamic = llm_dynamic

    # A. define the prompt template
    default_system = (
        "You are an AI assistant. You have access to tools, but you must NOT use them unless absolutely necessary.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. DO NOT use tools for creative writing (e.g., poems, stories, code), simple greetings, or general conversation. Answer these directly from your own knowledge.\n"
        "2. ONLY use retrieve_docs for specific questions about the company's internal knowledge and data.\n"
        "3. ONLY use web_search for specific questions about current events or external facts.\n"
        "4. If you have already used a tool and got a result, synthesize the final answer immediately. DO NOT call the tool again for the same question.\n"
        "5. Always provide concise and helpful answers."
    )
    
    if sys_prompt.strip():
        final_system = f"{default_system}\n\nUSER CUSTOM INSTRUCTIONS & PERSONA:\n{sys_prompt.strip()}"
    else:
        final_system = default_system

    print(f"--- DEBUG: System Prompt: {final_system} ---")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", final_system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Bind template with tools

    chain = prompt_template | llm_with_tools_dynamic


    # B. Generate response with Guardrails
    res = await chain.ainvoke({"messages": state["messages"]})

    status = "Finalizing Response..."

    # C. Run Guardrails on the output text ONLY if it's not a tool call
    if res.tool_calls:
        status = "Calling Tools..."
    else:
        status = "Answering..."

        if res.content.strip():

            # nemo_input = [
            #                 {
            #                     "role": "user", 
            #                     "content": f"Context: {context}\n\nQuestion: {state['messages'][-1].content}"
            #                 },
            #                 {
            #                     "role": "assistant",
            #                     "content": res.content
            #                 }
            #             ]


            # --- Guardrails output check only ---
            check_messages = [{"role": "assistant", "content": res.content}]
            rails_result = await rails.generate_async(
               messages=check_messages,
            options={
                "output_vars": True,
                "rails": ["output"] 
            }
            )
            new_content = res.content
            if hasattr(rails_result, "content"):
                new_content = rails_result.content
            elif hasattr(rails_result, "response") and isinstance(rails_result.response, list) and len(rails_result.response) > 0:
                first_response = rails_result.response[0]
                if isinstance(first_response, dict):
                    new_content = first_response.get("content", res.content)
                else:
                    new_content = res.content
                
            #----------------------DEBUG---------------------------------------
            print(f"DEBUG: Original AI Content: {res.content}")
            print(f"DEBUG: Guardrails Result: {new_content}")
            #------------------------------------------------------------------

            # If NeMo returned an actual modified response (like a refusal), we update the content.
            # We remove the strict equality block because Llama 3 sometimes hallucinates during the safety check.
            if new_content and str(new_content).strip() != "":
                # Only override if NeMo specifically triggered a refusal flow
                if "sorry" in str(new_content).lower() or "cannot answer" in str(new_content).lower():
                    # DO NOT mutate res.content so we don't poison the LLM's memory!
                    # Just update the status flag, and we'll intercept it in the frontend stream.
                    status = "Response blocked by safety/fact-check guardrails."
    return {"messages": [res], "status": status}

tools = [web_search, retrieve_docs]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)
# --- 4. GRAPH ORCHESTRATION ---

workflow = StateGraph(AgentState)

workflow.add_node("llm", call_model_node)
workflow.add_node("tools", tool_node)


workflow.add_edge(START, "llm")
workflow.add_conditional_edges(
    "llm",
    tools_condition,  # Checks if the LLM called a tool
)
workflow.add_edge("tools", "llm")
# --- 5. MEMORY (PostgreSQL Persistence) ---

DB_USER = os.environ.get("PGSQL_USERNAME")
DB_PASSWORD = os.environ.get("PGSQL_PASSWORD")
DB_HOST = os.environ.get("PGSQL_HOST", "localhost")
DB_PORT = os.environ.get("PGSQL_PORT", "5432")
DB_NAME = os.environ.get("PGSQL_NAME")

DB_URI = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode=require&channel_binding=require"
)

pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    global graph_app
    global pool  # added
    # Initialize checkpointer and setup tables

    # create pool here  ← changed
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=10,
        min_size=1,  # keeps one active connection
        timeout=10,
        kwargs={"autocommit": True},
    )
    async with pool:
        # Create thread_titles table if it doesn't exist
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "CREATE TABLE IF NOT EXISTS thread_titles (thread_id TEXT PRIMARY KEY, title TEXT)"
                )
        
        vector_db = get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        # Compile graph with the async checkpointer
        graph_app = workflow.compile(checkpointer=checkpointer)
        yield
# --- 6. API / FRONTEND CONNECTION (FastAPI) ---

api = FastAPI(lifespan=lifespan)


# --- PIP FREEZE ---
@api.get("/__freeze")
def freeze():
    import pkg_resources
    return sorted([str(d) for d in pkg_resources.working_set])
# -----------------

class RenameThreadRequest(BaseModel):
    title: str

class IngestRequest(BaseModel):
    url: str

@api.post("/ingest")
def ingest_endpoint(request: IngestRequest):
    try:
        from ingestion import ingest_web_url
        result = ingest_web_url(request.url)
        return {"status": "success", "message": result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/threads")
async def get_threads():
    if not pool:
        return {"threads": []}
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT c.thread_id, t.title 
                FROM (SELECT DISTINCT thread_id FROM checkpoints) c 
                LEFT JOIN thread_titles t ON c.thread_id = t.thread_id;
            """)
            rows = await cur.fetchall()
            return {"threads": [{"id": row[0], "title": row[1] if row[1] else f"Chat {row[0]}"} for row in rows]}

@api.put("/threads/{thread_id}/title")
async def rename_thread(thread_id: str, request: RenameThreadRequest):
    if not pool:
        return {"status": "error", "message": "No database connection"}
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO thread_titles (thread_id, title) 
                VALUES (%s, %s) 
                ON CONFLICT (thread_id) DO UPDATE SET title = EXCLUDED.title;
            """, (thread_id, request.title))
    return {"status": "success", "title": request.title}

@api.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    if not pool:
        return {"status": "error", "message": "No database connection"}
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
            await cur.execute("DELETE FROM thread_titles WHERE thread_id = %s", (thread_id,))
            try:
                await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
            except Exception:
                pass
    return {"status": "success"}

@api.get("/chat/{thread_id}/history")
async def get_chat_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph_app.aget_state(config)
    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({
                "id": str(id(msg)),
                "sender": "User",
                "type": "text",
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            if not getattr(msg, "tool_calls", None) and msg.content.strip():
                history.append({
                    "id": str(id(msg)),
                    "sender": "Meridian",
                    "type": "text",
                    "content": msg.content
                })
    return {"history": history}


api.add_middleware(
    CORSMiddleware,
    # allow_origins=[os.environ.get("frontendURL", "http://localhost:5173")],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.post("/chat")
async def chat_endpoint(
    user_id: str, 
    thread_id: str, 
    message: str, 
    temperature: float = 0.1,
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    system_prompt: str = "",
    web_search_enabled: bool = True,
    rag_enabled: bool = True
):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 7}
    input_data = {
        "messages": [HumanMessage(content=message)],
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "system_prompt": system_prompt,
        "web_search_enabled": web_search_enabled,
        "rag_enabled": rag_enabled
    }

    async def event_generator():
        yield "data: [STATUS] Answering...\n\n"  # endpoint should immediately send a ping to show it's alive
        try:
            async for event in graph_app.astream(
                input_data, config=config, stream_mode="updates"
            ):
                # 1. Handle Status Updates (from any node that provides them)
                # The 'event' dict will look like: {"retrieve": {"status": "...", "context": "..."}}
            
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict) and node_output.get("status"):
                        yield f"data: [STATUS] {node_output['status']}\n\n"

                    # 2. Handle the Final AI Message (specifically from the llm node)
                    if node_name == "llm" and isinstance(node_output, dict) and "messages" in node_output:
                        # node_output["messages"] only contains the NEW messages from this node
                        last_message = node_output["messages"][-1]

                        if node_output.get("status") == "Response blocked by safety/fact-check guardrails.":
                            yield "data: Response blocked by safety/fact-check guardrails.\n\n"
                        elif not getattr(last_message, "tool_calls", None) and last_message.content.strip():
                            import json
                            payload = json.dumps(last_message.content)
                            yield f"data: {payload}\n\n"
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc().replace('\n', ' | ')
            yield f"data: [ERROR] {str(e)} --- TRACEBACK: {tb_str}\n\n"
        yield "data: [DONE]\n\n"  # termination event



    return StreamingResponse(event_generator(), media_type="text/event-stream")
