import json
import hashlib
import base64
from datetime import datetime
from typing import Any, Dict, List
import streamlit as st

def save_session_state(key: str, value: Any):
    """Save data to Streamlit session state"""
    st.session_state[key] = value

def load_session_state(key: str, default: Any = None) -> Any:
    """Load data from Streamlit session state"""
    return st.session_state.get(key, default)

def format_timestamp(timestamp: str, format_str: str = "%B %d, %Y %I:%M %p") -> str:
    """Format ISO timestamp to readable string"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime(format_str)
    except:
        return timestamp

def generate_user_id(email: str) -> str:
    """Generate unique user ID from email"""
    return hashlib.sha256(email.encode()).hexdigest()[:16]

def emotion_to_emoji(emotion: str) -> str:
    """Convert emotion to emoji"""
    emoji_map = {
        "joy": "😊",
        "sadness": "😢",
        "anger": "😠",
        "fear": "😨",
        "surprise": "😲",
        "love": "❤️",
        "anxiety": "😰",
        "stress": "😫",
        "excitement": "🎉",
        "contentment": "😌",
        "neutral": "😐",
        "disgust": "🤢"
    }
    return emoji_map.get(emotion, "💭")

def emotion_to_color(emotion: str) -> str:
    """Convert emotion to CSS color"""
    color_map = {
        "joy": "#FFD700",      # Gold
        "sadness": "#4169E1",  # Royal Blue
        "anger": "#FF4500",    # Orange Red
        "fear": "#8A2BE2",     # Blue Violet
        "surprise": "#00CED1", # Dark Turquoise
        "love": "#FF69B4",     # Hot Pink
        "anxiety": "#8B4513",  # Saddle Brown
        "stress": "#A0522D",   # Sienna
        "excitement": "#32CD32", # Lime Green
        "contentment": "#90EE90", # Light Green
        "neutral": "#808080",  # Gray
        "disgust": "#556B2F"   # Dark Olive Green
    }
    return color_map.get(emotion, "#808080")

def format_memory_for_display(memory: Dict, max_length: int = 200) -> str:
    """Format memory for display in UI"""
    content = memory.get("content", "")
    
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    emotion = memory.get("emotion", "neutral")
    timestamp = memory.get("timestamp", "")
    
    if timestamp:
        formatted_time = format_timestamp(timestamp, "%b %d, %Y")
        return f"{emotion_to_emoji(emotion)} [{formatted_time}] {content}"
    
    return f"{emotion_to_emoji(emotion)} {content}"

def validate_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def calculate_sentiment_score(emotion_details: Dict) -> float:
    """Calculate sentiment score from emotion details"""
    if not emotion_details or "all_emotions" not in emotion_details:
        return 0.5  # Neutral
    
    emotions = emotion_details["all_emotions"]
    
    # Weight emotions by sentiment value
    sentiment_weights = {
        "joy": 1.0,
        "love": 1.0,
        "excitement": 0.9,
        "contentment": 0.8,
        "surprise": 0.3,
        "neutral": 0.5,
        "sadness": -0.8,
        "anger": -0.9,
        "fear": -0.7,
        "anxiety": -0.6,
        "stress": -0.5,
        "disgust": -0.9
    }
    
    total_weight = 0
    total_score = 0
    
    for emotion, confidence in emotions.items():
        weight = sentiment_weights.get(emotion, 0)
        total_score += weight * confidence
        total_weight += abs(weight) * confidence
    
    if total_weight > 0:
        normalized_score = (total_score / total_weight + 1) / 2  # Scale to 0-1
        return normalized_score
    
    return 0.5

def create_progress_bar(percentage: float, height: int = 20) -> str:
    """Create HTML progress bar"""
    color = "#4CAF50" if percentage > 0.6 else "#FF9800" if percentage > 0.3 else "#F44336"
    
    return f"""
    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; height: {height}px;">
        <div style="width: {percentage*100}%; background-color: {color}; height: 100%; border-radius: 10px; 
                    display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
            {percentage*100:.1f}%
        </div>
    </div>
    """

def get_greeting_based_on_time() -> str:
    """Get time-appropriate greeting"""
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 22:
        return "Good evening"
    else:
        return "Good night"
