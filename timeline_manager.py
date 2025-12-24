import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from collections import defaultdict

class TimelineManager:
    """Manage and visualize life timeline"""
    
    def __init__(self, memory_engine):
        self.memory_engine = memory_engine
    
    def get_timeline_data(self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> List[Dict]:
        """Get timeline data with emotional context"""
        memories = self.memory_engine.get_timeline(start_date, end_date)
        
        timeline_data = []
        for memory in memories:
            timeline_entry = {
                "id": memory.get("id"),
                "timestamp": memory.get("timestamp"),
                "date": memory.get("timestamp")[:10],  # Just date part
                "type": memory.get("type", "unknown"),
                "content": memory.get("content", "")[:100] + "..." if len(memory.get("content", "")) > 100 else memory.get("content", ""),
                "emotion": memory.get("emotion", "neutral"),
                "full_content": memory.get("content", ""),
                "response_style": memory.get("response_style", {}),
                "emotion_details": memory.get("emotion_details", {})
            }
            timeline_data.append(timeline_entry)
        
        return timeline_data
    
    def create_emotion_timeline_chart(self, timeline_data: List[Dict]):
        """Create interactive emotion timeline chart"""
        if not timeline_data:
            return None
        
        # Prepare data
        df = pd.DataFrame(timeline_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Emotion color mapping
        emotion_colors = {
            "joy": "#FFD700",      # Gold
            "sadness": "#4169E1",  # Royal Blue
            "anger": "#FF4500",    # Orange Red
            "fear": "#8A2BE2",     # Blue Violet
            "surprise": "#00CED1", # Dark Turquoise
            "love": "#FF69B4",     # Hot Pink
            "anxiety": "#8B4513",  # Saddle Brown
            "stress": "#A0522D",   # Sienna
            "neutral": "#808080",  # Gray
            "excitement": "#32CD32", # Lime Green
            "contentment": "#90EE90" # Light Green
        }
        
        # Create scatter plot
        fig = go.Figure()
        
        for emotion, color in emotion_colors.items():
            emotion_data = df[df['emotion'] == emotion]
            if not emotion_data.empty:
                fig.add_trace(go.Scatter(
                    x=emotion_data['date'],
                    y=[emotion] * len(emotion_data),
                    mode='markers',
                    name=emotion.capitalize(),
                    marker=dict(
                        color=color,
                        size=15,
                        symbol='circle',
                        line=dict(width=1, color='white')
                    ),
                    text=emotion_data['content'],
                    hovertemplate="<b>%{x}</b><br>Emotion: %{y}<br>Memory: %{text}<extra></extra>",
                    customdata=emotion_data[['id', 'full_content']].values
                ))
        
        # Update layout
        fig.update_layout(
            title="Emotional Timeline",
            xaxis_title="Date",
            yaxis_title="Emotion",
            hovermode='closest',
            showlegend=True,
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)'
        )
        
        return fig
    
    def get_emotion_statistics(self, timeline_data: List[Dict]) -> Dict:
        """Get emotion statistics and insights"""
        if not timeline_data:
            return {}
        
        emotion_count = defaultdict(int)
        emotion_by_type = defaultdict(lambda: defaultdict(int))
        emotion_dates = defaultdict(list)
        
        for entry in timeline_data:
            emotion = entry.get("emotion", "neutral")
            memory_type = entry.get("type", "unknown")
            date = entry.get("date")
            
            emotion_count[emotion] += 1
            emotion_by_type[memory_type][emotion] += 1
            emotion_dates[emotion].append(date)
        
        # Calculate insights
        total_memories = len(timeline_data)
        dominant_emotion = max(emotion_count.items(), key=lambda x: x[1])[0] if emotion_count else "neutral"
        
        # Find emotional patterns by day of week
        day_patterns = defaultdict(lambda: defaultdict(int))
        for entry in timeline_data:
            try:
                date_obj = datetime.strptime(entry['date'], '%Y-%m-%d')
                day_name = date_obj.strftime('%A')
                emotion = entry['emotion']
                day_patterns[day_name][emotion] += 1
            except:
                continue
        
        # Most emotional day
        most_emotional_day = None
        max_emotions = 0
        for day, emotions in day_patterns.items():
            total = sum(emotions.values())
            if total > max_emotions:
                max_emotions = total
                most_emotional_day = day
        
        return {
            "total_memories": total_memories,
            "emotion_distribution": dict(emotion_count),
            "dominant_emotion": dominant_emotion,
            "emotion_by_type": dict(emotion_by_type),
            "most_emotional_day": most_emotional_day,
            "emotional_diversity": len(emotion_count) / len(self.get_available_emotions()),
            "insights": self._generate_insights(emotion_count, timeline_data)
        }
    
    def get_available_emotions(self):
        """Return list of available emotions"""
        return ["joy", "sadness", "anger", "fear", "surprise", "love", 
                "anxiety", "stress", "neutral", "excitement", "contentment"]
    
    def _generate_insights(self, emotion_count: Dict, timeline_data: List[Dict]) -> List[str]:
        """Generate insights from emotion data"""
        insights = []
        
        total = sum(emotion_count.values())
        if total == 0:
            return ["No data available yet"]
        
        # Insight 1: Most common emotion
        if emotion_count:
            top_emotion, count = max(emotion_count.items(), key=lambda x: x[1])
            percentage = (count / total) * 100
            insights.append(f"You most frequently experience {top_emotion} ({percentage:.1f}% of memories)")
        
        # Insight 2: Emotional diversity
        unique_emotions = len(emotion_count)
        if unique_emotions >= 8:
            insights.append("You express a wide range of emotions, showing emotional diversity")
        elif unique_emotions <= 3:
            insights.append("Your emotional expressions tend to focus on a few core feelings")
        
        # Insight 3: Positive vs Negative balance
        positive_emotions = ["joy", "love", "excitement", "contentment", "surprise"]
        negative_emotions = ["sadness", "anger", "fear", "anxiety", "stress"]
        
        pos_count = sum(emotion_count.get(e, 0) for e in positive_emotions)
        neg_count = sum(emotion_count.get(e, 0) for e in negative_emotions)
        
        if pos_count > neg_count * 1.5:
            insights.append("Your memories lean toward positive experiences")
        elif neg_count > pos_count * 1.5:
            insights.append("You've been processing more challenging emotions lately")
        else:
            insights.append("You maintain a balanced emotional perspective")
        
        # Insight 4: Recent trend
        if len(timeline_data) >= 10:
            recent = timeline_data[-10:]
            recent_emotions = [e.get("emotion") for e in recent]
            recent_pos = sum(1 for e in recent_emotions if e in positive_emotions)
            recent_neg = sum(1 for e in recent_emotions if e in negative_emotions)
            
            if recent_pos > recent_neg:
                insights.append("Recently, you've been in a more positive emotional space")
            elif recent_neg > recent_pos:
                insights.append("Lately, you've been working through more complex emotions")
        
        return insights[:3]  # Return top 3 insights
