import os
import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

dotenv.load_dotenv()
api_key = os.getenv("api_key")

# In-memory conversation history for the hardcoded thread
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

# Simple in-memory app without checkpointer
app = workflow.compile()

def chat_with_bot(user_message: str, session_id: str = None):
    """Chat with bot using in-memory conversation history with hardcoded thread 'abc123'"""
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
        
        # Get response from model
        result = app.invoke({"messages": messages}, config={})
        
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