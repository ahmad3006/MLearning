import os
import dotenv
import requests
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

dotenv.load_dotenv()
api_key = os.getenv("api_key")
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback in-memory conversation history (used if database fails)
conversation_history = []

# Weather API configuration
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast?latitude=-7.9797&longitude=112.6304&hourly=temperature_2m,shortwave_radiation&timezone=auto&forecast_days=1"

def fetch_weather_data():
    """Fetch current weather data from Open-Meteo API"""
    try:
        response = requests.get(WEATHER_API_URL)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# Change location and hour based on API data
def format_weather_data(weather_data):
    """Format weather data into readable text for RAG"""
    if not weather_data:
        return "Weather data is currently unavailable."
    
    try:
        # Extract current time and find the closest hour index
        current_time = datetime.now()
        hourly_data = weather_data.get('hourly', {})
        times = hourly_data.get('time', [])
        temperatures = hourly_data.get('temperature_2m', [])
        solar_radiation = hourly_data.get('shortwave_radiation', [])
        
        # Find current hour index (simplified - takes first available data)
        current_temp = temperatures[0] if temperatures else "N/A"
        current_solar = solar_radiation[0] if solar_radiation else "N/A"
        
        # Get min/max temperatures for the day
        min_temp = min(temperatures) if temperatures else "N/A"
        max_temp = max(temperatures) if temperatures else "N/A"
        avg_solar = sum(solar_radiation) / len(solar_radiation) if solar_radiation else "N/A"
        
        formatted_data = f"""
CURRENT WEATHER DATA:
Location: Latitude -7.9797, Longitude 112.6304 (Malang, Indonesia region)
Current Temperature: {current_temp}°C
Current Solar Radiation: {current_solar} W/m²

TODAY'S FORECAST:
Minimum Temperature: {min_temp}°C
Maximum Temperature: {max_temp}°C
Average Solar Radiation: {avg_solar:.2f} W/m²

SOLAR POWER INSIGHTS:
- Solar radiation levels indicate the potential for solar power generation
- Higher radiation values (>500 W/m²) are excellent for solar panels
- Temperature affects solar panel efficiency (optimal around 25°C)
- Current conditions: {"Excellent" if float(current_solar) > 500 else "Good" if float(current_solar) > 200 else "Poor"} for solar power generation

HOURLY DATA AVAILABLE:
Times: {len(times)} hours of data
Temperature range: {min_temp}°C to {max_temp}°C
Solar radiation data: Available for all hours
"""
        return formatted_data
    except Exception as e:
        return f"Error formatting weather data: {e}"

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

# Load and process documents for RAG - now using dynamic weather data
def create_weather_documents():
    """Create documents from current weather data"""
    weather_data = fetch_weather_data()
    formatted_weather = format_weather_data(weather_data)
    
    # Also load static FAQ if it exists
    static_content = ""
    try:
        loader = TextLoader("static-faq.txt", encoding="utf-8")
        static_docs = loader.load()
        static_content = "\n".join([doc.page_content for doc in static_docs])
    except Exception as e:
        print(f"Could not load static FAQ: {e}")
    
    # Combine weather data with static content
    combined_content = f"{formatted_weather}\n\n{static_content}"
    
    # Create document
    weather_doc = Document(
        page_content=combined_content,
        metadata={"source": "weather_api", "timestamp": datetime.now().isoformat()}
    )
    
    return [weather_doc]

# Create dynamic documents
docs = create_weather_documents()
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
# rag_prompt = hub.pull("rlm/rag-prompt")

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant specializing in solar power systems. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, provide a general helpful response."),
#     MessagesPlaceholder(variable_name="messages"),
# ])

model = llm

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    # Get the latest user message for RAG retrieval
    latest_message = state["messages"][-1].content if state["messages"] else ""
    
    # Refresh weather data for each query to get latest information
    fresh_docs = create_weather_documents()
    fresh_splits = text_splitter.split_documents(fresh_docs)
    
    # Update vector store with fresh data
    try:
        # Clear existing documents and add fresh ones
        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(documents=fresh_splits)
    except Exception as e:
        print(f"Error updating vector store: {e}")
        # Fallback to existing vector store
        pass
    
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(latest_message, k=3)
    
    # Format context from retrieved documents
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Create enhanced system message with context
    enhanced_system_message = f"""You are a helpful assistant specializing in solar power systems and weather analysis. 

You have access to real-time weather data including temperature and solar radiation measurements from Malang, Indonesia region (Latitude -7.9797, Longitude 112.6304).

Use the following context to answer the user's question:

{context}

Key capabilities:
- Provide current weather conditions and forecasts
- Analyze solar radiation levels for solar power generation potential
- Give advice on solar panel efficiency based on current conditions
- Answer questions about weather patterns and their impact on solar energy
- Provide both current readings and daily forecasts

If the user asks about weather, solar conditions, or energy generation, use the real-time data provided. For general solar power questions, use both the weather data and your knowledge to provide comprehensive answers."""
    
    # Convert to proper message format for Azure OpenAI
    formatted_messages = [SystemMessage(content=enhanced_system_message)]
    
    # Add the conversation messages
    for msg in state["messages"]:
        formatted_messages.append(msg)
    
    # Get response from LLM
    response = model.invoke(formatted_messages)
    return {"messages": [response]}

# Graph structure
# masih pakai yang dynamic, belum dynamic + static
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
