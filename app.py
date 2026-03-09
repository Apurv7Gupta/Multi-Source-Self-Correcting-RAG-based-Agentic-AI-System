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
from langchain_community.tools.tavily_search import TavilySearchResults
from nemoguardrails import LLMRails, RailsConfig
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
from db_config import get_vector_db
# --- 1. LLM CONFIGURATION ---
llm_tool_error = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.1,
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
    context: str
    status: str
# --- 3. NODES / AGENT LOGIC ---


# 1). INITIALIZE GUARDRAILS ---
config = RailsConfig.from_path("./config")
rails = LLMRails(config, llm=llm)  # Pass existing LLM


# 1.5) SEARCH THE WEB (tool) ---

search_tool = TavilySearchResults(k=3)


@tool
async def web_search(query: str):
    """
    Search the web for real-time information or topics not found in the internal Docs.
    Use this only when the user asks about current events or specific external data, or something you really don't know
    """
    results = await search_tool.ainvoke(query)

    joined_results = "\n".join([r["content"] for r in results])
    return f"--------- WEB SEARCH RESULTS ---------\n\n{joined_results}"


# 2). RETRIEVE NODE ---


async def retrieve_node(state: AgentState):
    """Fetch relevant documents + web search based on the last user query."""
    last_message = state["messages"][-1].content

    joined_results = ""

    if retriever is None:
        docs_task = asyncio.sleep(0, result=[])
    else:
        docs_task = retriever.ainvoke(last_message)

    if not isinstance(docs_task, Exception):
        joined_results = "\n".join([doc.page_content for doc in docs_task])

    docs_context = f"--- INTERNAL DOCS KNOWLEDGE ---\n{joined_results}\n\n"

    return {
        "context": docs_context,
        "status": "Scanning docs...",
    }


#  3). CALL MODEL NODE WITH GUARDRAILS & PROMPT TEMPLATE ---
async def call_model_node(state: AgentState):
    context = state["context"]

    # A. define the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer concisely based on the provided context ONLY if you think the question is related to the context, otherwise you can answer based on your own knowledge, but do NOT hallucinate) "
                "\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Bind template with tools

    chain = prompt_template | llm_with_tools

    # B. Generate response with Guardrails
    res = await chain.ainvoke({"context": context, "messages": state["messages"]})

    status = "Processing..."

    # C. Run Guardrails on the output text ONLY if it's not a tool call
    if not res.tool_calls:

        if res.content.strip():

            nemo_input = [
                            {
                                "role": "user", 
                                "content": f"Context: {context}\n\nQuestion: {state['messages'][-1].content}"
                            },
                            {
                                "role": "assistant",
                                "content": res.content
                            }
                        ]

            result = await rails.generate_async(messages=nemo_input)
            res.content = result.content
            status = "Finalizing response..."
        if not res.content or "I cannot answer" in res.content:
            status = "Response blocked by safety/fact-check guardrails."

    else:
        status = "Searching the web for more info..."

    return {"messages": [res], "status": status}


tools = [web_search]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)
# --- 4. GRAPH ORCHESTRATION ---

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("llm", call_model_node)
workflow.add_node("tools", tool_node)


workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "llm")
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

pool = AsyncConnectionPool(conninfo=DB_URI, max_size=10, kwargs={"autocommit": True})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    global graph_app
    # Initialize checkpointer and setup tables
    async with pool:
        vector_db = get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        # Compile graph with the async checkpointer
        graph_app = workflow.compile(checkpointer=checkpointer)
        yield
        await pool.close()
# --- 6. API / FRONTEND CONNECTION (FastAPI) ---

api = FastAPI(lifespan=lifespan)


@api.post("/chat")
async def chat_endpoint(user_id: str, thread_id: str, message: str):
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=message)]}

    async def event_generator():
        async for event in graph_app.astream(
            input_data, config=config, stream_mode="updates"
        ):
            # 1. Handle Status Updates (from any node that provides them)
            # The 'event' dict will look like: {"retrieve": {"status": "...", "context": "..."}}
            node_name = list(event.keys())[0]
            node_output = event[node_name]

            for node_name, node_output in event.items():
                if "status" in node_output:
                    yield f"data: [STATUS] {node_output['status']}\n\n"

                # 2. Handle the Final AI Message (specifically from the llm node)
                if node_name == "llm" and "messages" in node_output:
                    # node_output["messages"] only contains the NEW messages from this node
                    final_content = node_output["messages"][-1].content

                    if final_content.strip():
                        yield f"data: {final_content}\n\n"



    return StreamingResponse(event_generator(), media_type="text/event-stream")
