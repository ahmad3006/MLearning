from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import joblib
from chatbot import chat_with_bot
import requests

app = FastAPI()

# Allow frontend access (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model once at startup
model = joblib.load("solar_power_prediction.pkl")


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
        total_energy = sum(row["predicted_power"] for row in results)
        # Save results
        results.append({
            "time": time,
            "ambient_temperature": round(ambient_temperature, 2),
            "module_temperature": round(module_temp, 2),
            "irradiance": round(irradiance, 2),
            "predicted_power": round(predicted_power, 2)
        })

    return {"data": results, "total_energy": round(total_energy, 2)}

@app.post("/chat")
async def chat(request: Request):
    try:
        # Check if request has content
        body_bytes = await request.body()
        if not body_bytes:
            return {"error": "Request body is empty"}
        
        # Parse JSON
        import json
        body = json.loads(body_bytes.decode('utf-8'))
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON in request body", "details": str(e)}
    except Exception as e:
        return {"error": "Failed to process request body", "details": str(e)}
    
    message = body.get("message")
    session_id = body.get("session_id")  # Optional session ID
    
    if not message:
        return {"error": "Message field is required"}
    
    if not isinstance(message, str):
        return {"error": "Message must be a string"}

    try:
        response = chat_with_bot(message, session_id)
        return response  # This now includes both response and session_id
    except Exception as e:
        return {"error": "Error processing chat request", "details": str(e)}
