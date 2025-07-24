# app.py
"""
Simple FastAPI backend for Mental Health Support Chatbot
Loads fine-tuned DistilBERT and exposes a /chat endpoint
"""
import os
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from fastapi.middleware.cors import CORSMiddleware
import threading

# Load model and tokenizer
MODEL_PATH = "model_weights"  # Change if you save elsewhere
EMOTIONS = ["anxiety", "depression", "stress", "hope", "crisis"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading with better error handling
try:
    if not os.path.exists("model_weights"):
        raise FileNotFoundError("model_weights directory not found")
    tokenizer = DistilBertTokenizerFast.from_pretrained("model_weights", local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained("model_weights", local_files_only=True)
    model.eval()
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise RuntimeError("Failed to load model. Please train first.")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    emotion: str
    response: str

RESPONSES = {
    "anxiety": "I'm here for you. Can you tell me more about what's making you anxious?",
    "depression": "I'm sorry you're feeling this way. You're not alone.",
    "stress": "Stress can be overwhelming. What has been causing you stress lately?",
    "hope": "That's wonderful! I'm glad you're feeling hopeful. What are you looking forward to?",
    "crisis": "If you're in crisis, please reach out to a professional or call a helpline."
}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        inputs = tokenizer(
            req.message,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        emotion = EMOTIONS[pred]
        response = RESPONSES.get(emotion, "I'm here to help.")
        return ChatResponse(emotion=emotion, response=response)
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing message")

# Flask app for serving frontend and backend
flask_app = Flask(__name__, static_folder='.', template_folder='.')

# Serve index.html
@flask_app.route('/')
def index():
    with open('index.html', 'r') as f:
        return f.read()

# Serve favicon.ico (optional)
@flask_app.route('/favicon.ico')
def favicon():
    return '', 204

# Chat API endpoint
@flask_app.route('/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    message = data.get('message', '').lower()
    # Simple keyword-based override for positive/happy messages
    if any(word in message for word in ["happy", "joy", "excited", "good", "great", "hopeful"]):
        return jsonify({"emotion": "hope", "response": RESPONSES["hope"]})
    try:
        inputs = tokenizer(message, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        emotion = EMOTIONS[pred]
        response = RESPONSES.get(emotion, "I'm here to help.")
        return jsonify({"emotion": emotion, "response": response})
    except Exception as e:
        return jsonify({"emotion": "error", "response": "Sorry, something went wrong."}), 500

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=8000, debug=True)
