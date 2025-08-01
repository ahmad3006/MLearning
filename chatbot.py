import os
import dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


dotenv.load_dotenv()
api_key = os.getenv("api_key")
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback in-memory conversation history (used if database fails)
conversation_history = []

# Initialize Azure LLM and embeddings for RAG
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("azure_gpt-3.5"),
    azure_deployment="gpt-35-turbo",
    api_key=os.getenv("azure_key"),
    api_version="2024-12-01-preview",
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("azure_embed_url"),
    azure_deployment="text-embedding-3-small",
    api_key=os.getenv("azure_key"),
    api_version="2024-02-01",
)

# Load and process documents for RAG
loader = TextLoader("static-faq.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n---\n\n", "\n\n", "\n", " "]
)
all_splits = text_splitter.split_documents(docs)

# Create vector store
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents=all_splits)

# Get RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in solar power systems. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, provide a general helpful response."),
    MessagesPlaceholder(variable_name="messages"),
])

model = llm

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    # Get the latest user message for RAG retrieval
    latest_message = state["messages"][-1].content if state["messages"] else ""
    
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(latest_message, k=3)
    
    # Format context from retrieved documents
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Create enhanced system message with context
    enhanced_system_message = f"""You are a helpful assistant specializing in solar power systems. 

Use the following context to answer the user's question:

{context}

If the context contains relevant information, use it to provide an accurate answer. If the context doesn't contain relevant information, provide a general helpful response based on your knowledge."""
    
    # Convert to proper message format for Azure OpenAI
    formatted_messages = [SystemMessage(content=enhanced_system_message)]
    
    # Add the conversation messages
    for msg in state["messages"]:
        formatted_messages.append(msg)
    
    # Get response from LLM
    response = model.invoke(formatted_messages)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Initialize PostgreSQL checkpointer with error handling
checkpointer = None
USE_DB = False
app = None

if DATABASE_URL:
    try:
        # Simple synchronous connection approach
        import psycopg
        
        # Create a connection
        conn = psycopg.connect(DATABASE_URL)
        
        # Create PostgresSaver with the connection
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()  # Create tables if they don't exist
        
        app = workflow.compile(checkpointer=checkpointer)
        print("Database connection successful - using PostgreSQL persistence")
        USE_DB = True
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Falling back to in-memory storage")
        app = workflow.compile()  # No checkpointer - non-persistent
        USE_DB = False
else:
    print("No DATABASE_URL provided, using in-memory storage")
    app = workflow.compile()  # No checkpointer - non-persistent
    USE_DB = False

def chat_with_bot(user_message: str, session_id: str = None):
    """Chat with bot using PostgreSQL persistence with hardcoded thread 'abc123'"""
    # Use hardcoded thread ID
    thread_id = "abc123"
    config = {"configurable": {"thread_id": thread_id}}
    
    if USE_DB and app:
        # Try with PostgreSQL database first
        try:
            input_messages = [HumanMessage(content=user_message)]
            result = app.invoke({"messages": input_messages}, config=config)
            return {
                "response": result["messages"][-1].content,
                "session_id": "abc123"  # Hardcoded thread ID
            }
        except Exception as e:
            print(f"Database error: {e}")
            print("Falling back to in-memory conversation")
    
    # Fallback: Use in-memory conversation history
    try:
        # Build conversation from history
        messages = []
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current message
        messages.append(HumanMessage(content=user_message))
        
        # Create a temporary app without checkpointer for fallback
        temp_app = workflow.compile()
        result = temp_app.invoke({"messages": messages}, config={})
        
        # Store in conversation history
        conversation_history.append({
            "role": "user", 
            "content": user_message
        })
        conversation_history.append({
            "role": "assistant", 
            "content": result["messages"][-1].content
        })
        
        return {
            "response": result["messages"][-1].content,
            "session_id": "abc123"  # Hardcoded thread ID
        }
    except Exception as e:
        raise Exception(f"Chat failed: {e}")