import logging
import os
from typing import Optional, Dict, Any, List
# from langchain_community.embeddings import HuggingFaceEmbeddings
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Model Initialization

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables.") 

llm = ChatGroq(
    model= os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
    temperature=os.getenv("LLM_TEMPERATURE", "0.0"),
    )


# Vector Store (RAG setup)

EMBEDDINGS_MODEL = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_db")

vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=EMBEDDINGS_MODEL
)

# Sample Data

# Initialize Chroma vector store
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_db")
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=EMBEDDINGS_MODEL
)

# Sample data to store in Chroma DB
sample_data = [
    {"id": "1", "text": "To pay bill contact account department", "metadata": {"category": "account"}},
    {"id": "2", "text": "Leave application must be approved by HR over email", "metadata": {"category": "HR"}},
    {"id": "3", "text": "Contact IT Support for technical issues", "metadata": {"category": "technical"}},
    {"id": "4", "text": "Get in touch with the help desk on 1st floor", "metadata": {"category": "general"}},
]

# Add data to Chroma DB
for item in sample_data:
    vectorstore.add_texts(
        texts=[item["text"]],
        metadatas=[item["metadata"]],
        ids=[item["id"]]
    )

# Persist the data to disk
vectorstore.persist()

print("Sample data has been added to Chroma DB.")



retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# Sate Definition
VALID_INTENT = {"billing", "technical", "account","general"}

class ChatState(TypedDict, total=False):
    message: str
    intent: Optional[str]
    context: Optional[str]
    response: Optional[str]

# Extract text safely from LLM Result

def _extract_text_from_result(res: Any) -> str:
    if res is None:
        return ""

    if hasattr(res, "content"):
        return (res.content or "").strip()

    if hasattr(res, "text"):
        return (res.text or "").strip()
    
    try:
        d = res.dict()
        if isinstance(d, dict):
            return (d.get("content") or d.get("text") or "").strip()
    except Exception:
            pass
    
    return str(res).strip()

async def _llm_ainvoke_safe(prompt: str) -> str:

    if hasattr(llm, "ainvoke"): #ainovke Runnable (Chains, agentx, models)
        try:
            res = await llm.ainvoke(prompt)
            return _extract_text_from_result(res)
        except Exception:
            logger.exception("ainvoke failed")

    if hasattr(llm, "apredict"): #Legacy method for getting simple prection
        try:
            res = await llm.apredict(prompt)
            if isinstance(res, str):
                return res.strip()
            return _extract_text_from_result(res)
        except Exception:
            logger.exception("apredict failed")

    if hasattr(llm, "invoke"):
        try:
            res = llm.invoke(prompt)
            return _extract_text_from_result(res)
        except Exception:
            logger.exception("invoke failed")   
    
    return ""

# Node 1 : Intent Classification

async def intent_node(state: ChatState) -> Dict[str, str]:
    message = state.get("message", "").strip()
    if not message:
        raise ValueError("EMpty message reciebed in intent node")

    prompt = (
        "You are an intent classifier for a SaaS support system.\n"
        "Classify the user message into exactly one of:\n"
        "billing, technical, account, general.\n\n"
        "Return ONLY the single word category.\n\n"
        f"User message:\n{message}\n\nCategory:"
    )

    raw = await _llm_ainvoke_safe(prompt)
    intent = raw.lower().strip()

    if intent not in VALID_INTENT:
        intent = "general"

    logger.info("Intenet classified: %s", intent)

    return {"intent": intent}

# Node 2: Retrieval RAG

async def retrieve_node(state: ChatState) -> Dict[str, str]:
    message = state.get("message", "").strip()

    if not message:
        return {"context": ""}

    try:
        # Pass `run_manager=None` as the required keyword argument
        docs = retriever._get_relevant_documents(message, run_manager=None)  # Retrieval
        context_chunks: List[str] = [doc.page_content for doc in docs]
        context = "\n\n".join(context_chunks)
        logger.info("Retrieved documents: %s", context_chunks)
        logger.info("Retrieved %d context documents", len(context_chunks))
        return {"context": context}
    except Exception:
        logger.exception("Retrieval failed")
        return {"context": ""}

# Node 3:  Response generation

async def response_node(state: ChatState) -> Dict[str, str]:
    message = state.get("message", "").strip()
    intent = state.get("intent", "general")
    context = state.get("context", "")

    if not message:
        return {"response": "Sorry, I didn't receive any message."}

    if intent == "general" and context:
        # Use only the most relevant document for the "general" intent
        context_lines = context.split("\n\n")
        most_relevant_context = context_lines[0] if context_lines else ""
        logger.info("Using most relevant context for general intent: %s", most_relevant_context)
        return {"response": most_relevant_context}

    prompt = (
        "You are a professional SaaS support assistant.\n\n"
        "Use the following retrieved context to answer the user's query.\n"
        "If the context does not contain the answer, say:\n"
        "'I couldn't find this in our documentation. Let me connect you with support.'\n\n"
        "============================\n"
        "Retrieved Context:\n"
        f"{context}\n"
        "============================\n\n"
        f"User intent: {intent}\n"
        f"User message: {message}\n\n"
        "Provide a concise, accurate, actionable answer based on the retrieved context."
    )

    raw = await _llm_ainvoke_safe(prompt)
    response = raw.strip()
    logger.info("Generated response of length %d", len(response))
    return {"response": response}

# Node 4: Safety Layer

def safety_node(state: ChatState) -> Dict[str, str]:
    response = state.get("response", "").strip()

    if not response:
        return {"response": "Sorry, I couldn't generate a response."}

    # Simple safety check for sensitive info (can be expanded with more complex logic)
    restricted_keywords = ["diagnose", "prescribe", "legal advice", "medical advice", "lawyer", "doctor"]

    if any(keyword in response.lower() for keyword in restricted_keywords):
        return {
            "response": (
                "I'm unable to provide medical or legal advice. "
                "Please consult a qualified professional."
            )
        }

    return {"response": response}

# Build Langgraph workflow

graph = StateGraph(ChatState)

graph.add_node("intent", intent_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("response", response_node)
graph.add_node("safety", safety_node)

graph.set_entry_point("intent")

graph.add_edge("intent", "retrieve")
graph.add_edge("retrieve", "response")
graph.add_edge("response", "safety")

graph.set_finish_point("safety")

app_graph = graph.compile()



