from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI
from datetime import datetime, timezone
from fastapi import HTTPException
from pydantic import BaseModel, Field


from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1")
 

app = FastAPI()

# In-memory session store
chat_sessions = {}


def build_prompt(session_history, user_message):

    system_instruction = "You are a helpful AI assistant."

    history_text = ""

    for msg in session_history:
        role = msg["role"]
        content = msg["content"]
        history_text += f"{role.capitalize()}: {content}\n"

    prompt = f"""
{system_instruction}

Conversation history:
{history_text}

User: {user_message}
Assistant:
"""

    return prompt.strip()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    timestamp: str



@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        session_id = request.session_id
        message = request.message.strip()

        # Business validation
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        #Create session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        session_history = chat_sessions[session_id]

        #Build prompt
        prompt = build_prompt(session_history, message)

        response = client.responses.create(
        model="mistralai/mistral-7b-instruct",
        input=prompt
        )

        reply_text = response.output_text


        #Store messages
        session_history.append({
            "role": "user",
            "content": message
        })

        session_history.append({
            "role": "assistant",
            "content": reply_text
        })

        return ChatResponse(
            reply=reply_text,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
