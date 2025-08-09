from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage
import requests
import joblib
from chatbotengine import chatbot_app  # updated import

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load prediction model
model = joblib.load("solar_power_prediction.pkl")

# Request body model
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default_thread"

@app.post("/chat")
def run_chatbot(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    input_messages = [HumanMessage(content=request.message)]
    
    response_state = chatbot_app.invoke({"messages": input_messages}, config)
    return {"thread_id": request.thread_id, "response": response_state["messages"][-1].content}


@app.get("/predict")
def predict():
    url = "https://api.open-meteo.com/v1/forecast?latitude=-7.9797&longitude=112.6304&hourly=temperature_2m,shortwave_radiation&timezone=auto&forecast_days=1"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Failed to fetch weather data"}

    data = response.json()
    hourly = data["hourly"]
    
    results = []

    for i in range(24):
        time = hourly["time"][i]
        ambient_temperature = hourly["temperature_2m"][i]
        irradiance = hourly["shortwave_radiation"][i]
        module_temp = ambient_temperature + irradiance * 0.03 # NOCT Prediction

        # Prepare model input
        input_vector = [[ambient_temperature, module_temp, irradiance]]
        predicted_power = model.predict(input_vector)[0]
        
        # Save results
        results.append({
            "time": time,
            "ambient_temperature": round(ambient_temperature, 2),
            "module_temperature": round(module_temp, 2),
            "irradiance": round(irradiance, 2),
            "predicted_power": round(predicted_power, 2)
        })

    # Calculate total energy after all results are collected
    total_energy = sum(row["predicted_power"] for row in results)
    return {"data": results, "total_energy": round(total_energy, 2)}
