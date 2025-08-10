from langchain.chat_models import init_chat_model
import os
import dotenv

dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# print("Google API Key:", GOOGLE_API_KEY)
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=GOOGLE_API_KEY)
response = model.invoke("Hello, World!")
print(response)