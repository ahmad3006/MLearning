import os
import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

dotenv.load_dotenv()
api_key = os.getenv("api_key")
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback in-memory conversation history (used if database fails)
conversation_history = []

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant"),
    MessagesPlaceholder(variable_name="messages"),
])

model = init_chat_model(
    model="gpt-3.5-turbo",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

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