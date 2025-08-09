import os
import dotenv
import requests
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,MessagesState, StateGraph, END


dotenv.load_dotenv()
api_key = os.getenv("api_key")
# DATABASE_URL = os.getenv("DATABASE_URL")

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
    azure_endpoint=os.getenv("AZURE_3_5"),
    azure_deployment="gpt-35-turbo",
    api_key=os.getenv("AZURE_KEY"),
    api_version="2024-12-01-preview",
)


# 1. Weather context node
def add_weather_context(state: MessagesState) -> MessagesState:
    weather_data = fetch_weather_data()
    formatted_weather = format_weather_data(weather_data)
    
    # Insert as a system message at the start
    state["messages"].insert(
        0,
        SystemMessage(content=f"Current weather data:\n{formatted_weather}")
    )
    return state

# 2. Model call node
def call_model(state: MessagesState) -> MessagesState:
    # Append the model's reply instead of overwriting the list
    response = model.invoke(state["messages"])
    state["messages"].append(response)
    return state

# 3. Build the graph
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("add_weather", add_weather_context)
graph.add_node("model", call_model)

# Define flow: START → add_weather → model → END
graph.add_edge(START, "add_weather")
graph.add_edge("add_weather", "model")
graph.add_edge("model", END)

# Compile with memory
memory = MemorySaver()
chatbot_app = graph.compile(checkpointer=memory)

