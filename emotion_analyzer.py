from typing import Dict, List, Tuple
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import numpy as np

from config import settings

class EmotionAnalyzer:
    """Analyze emotions from text and voice"""
    
    def __init__(self):
        # Text emotion analysis
        self.text_emotion_model = pipeline(
            "text-classification",
            model=settings.EMOTION_MODEL,
            return_all_scores=True
        )
        
        # Emotion categories
        self.emotion_categories = [
            "joy", "sadness", "anger", "fear", 
            "surprise", "disgust", "neutral",
            "excitement", "anxiety", "stress", 
            "contentment", "love", "nostalgia"
        ]
        
        # Voice emotion model (placeholder - would need actual voice analysis)
        self.voice_emotion_model = None
        
    def analyze_text(self, text: str) -> Dict:
        """Analyze emotion from text"""
        if not text.strip():
            return {"dominant_emotion": "neutral", "confidence": 1.0, "all_emotions": {}}
        
        try:
            results = self.text_emotion_model(text)[0]
            
            # Map to our emotion categories
            emotion_scores = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                # Map similar emotions
                if label in ['joy', 'happy']:
                    emotion_scores['joy'] = emotion_scores.get('joy', 0) + score
                elif label in ['sadness', 'sad']:
                    emotion_scores['sadness'] = emotion_scores.get('sadness', 0) + score
                elif label == 'anger':
                    emotion_scores['anger'] = score
                elif label == 'fear':
                    emotion_scores['fear'] = score
                elif label == 'surprise':
                    emotion_scores['surprise'] = score
                elif label == 'disgust':
                    emotion_scores['disgust'] = score
                elif 'anxiety' in label or 'nervous' in label:
                    emotion_scores['anxiety'] = emotion_scores.get('anxiety', 0) + score
                elif 'love' in label or 'affection' in label:
                    emotion_scores['love'] = emotion_scores.get('love', 0) + score
            
            # Normalize scores
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v/total for k, v in emotion_scores.items()}
            
            # Get dominant emotion
            if emotion_scores:
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                return {
                    "dominant_emotion": dominant_emotion[0],
                    "confidence": dominant_emotion[1],
                    "all_emotions": emotion_scores,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
        
        return {"dominant_emotion": "neutral", "confidence": 1.0, "all_emotions": {}}
    
    def analyze_conversation_pattern(self, messages: List[Dict]) -> Dict:
        """Analyze conversation patterns and emotional trends"""
        if not messages:
            return {"mood_trend": "stable", "emotional_variety": 0, "dominant_pattern": "neutral"}
        
        emotions = []
        for msg in messages:
            if msg.get('emotion'):
                emotions.append(msg['emotion'])
        
        if not emotions:
            return {"mood_trend": "stable", "emotional_variety": 0, "dominant_pattern": "neutral"}
        
        # Calculate emotional variety
        unique_emotions = len(set(emotions))
        emotional_variety = unique_emotions / len(self.emotion_categories)
        
        # Determine mood trend (last 5 messages)
        recent_emotions = emotions[-5:]
        positive_emotions = ['joy', 'excitement', 'love', 'contentment', 'surprise']
        negative_emotions = ['sadness', 'anger', 'fear', 'anxiety', 'stress', 'disgust']
        
        pos_count = sum(1 for e in recent_emotions if e in positive_emotions)
        neg_count = sum(1 for e in recent_emotions if e in negative_emotions)
        
        if pos_count > neg_count:
            mood_trend = "positive"
        elif neg_count > pos_count:
            mood_trend = "negative"
        else:
            mood_trend = "stable"
        
        # Find most common emotion pattern
        from collections import Counter
        emotion_counter = Counter(emotions)
        dominant_pattern = emotion_counter.most_common(1)[0][0] if emotion_counter else "neutral"
        
        return {
            "mood_trend": mood_trend,
            "emotional_variety": emotional_variety,
            "dominant_pattern": dominant_pattern,
            "emotion_history": emotions[-10:]  # Last 10 emotions
        }
    
    def get_emotional_response_style(self, user_emotion: str, confidence: float) -> Dict:
        """Determine appropriate response style based on user emotion"""
        response_styles = {
            "joy": {
                "tone": "enthusiastic",
                "response_length": "medium",
                "emoji_frequency": "high",
                "empathy_level": "celebratory"
            },
            "sadness": {
                "tone": "gentle",
                "response_length": "longer",
                "emoji_frequency": "low",
                "empathy_level": "high"
            },
            "anxiety": {
                "tone": "calm",
                "response_length": "medium",
                "emoji_frequency": "medium",
                "empathy_level": "reassuring"
            },
            "anger": {
                "tone": "neutral",
                "response_length": "shorter",
                "emoji_frequency": "none",
                "empathy_level": "understanding"
            },
            "love": {
                "tone": "warm",
                "response_length": "medium",
                "emoji_frequency": "high",
                "empathy_level": "reciprocal"
            },
            "neutral": {
                "tone": "balanced",
                "response_length": "medium",
                "emoji_frequency": "medium",
                "empathy_level": "normal"
            }
        }
        
        return response_styles.get(user_emotion, response_styles["neutral"])
