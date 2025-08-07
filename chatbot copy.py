import os
import dotenv
import requests
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.checkpoint.postgres import PostgresSaver

dotenv.load_dotenv()
api_key = os.getenv("api_key")
DATABASE_URL = os.getenv("DATABASE_URL")

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
model = AzureChatOpenAI(
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

# model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )

from langgraph.graph import START,MessagesState, StateGraph

# Memory Persistence
with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
    # checkpointer.setup()

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # for chunk in graph.stream(
    #     {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    #     config,
    #     stream_mode="values"
    # ):
    #     chunk["messages"][-1].pretty_print()

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()