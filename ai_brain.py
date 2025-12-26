import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os

from config import settings
from memory_engine import MemoryEngine
from emotion_analyzer import EmotionAnalyzer

class EchoSoulAI:
    """Core AI Brain for EchoSoul with Gemini"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_engine = MemoryEngine(user_id)
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Initialize conversation memory
        self.conversation_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        
        # Load personality
        self.personality = self._load_personality()
        
        # Initialize Gemini LLM via LangChain
        if settings.USE_GEMINI and settings.GOOGLE_API_KEY:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                temperature=0.7,
                google_api_key=settings.GOOGLE_API_KEY
            )
            self.use_gemini = True
        else:
            # Fallback to local model
            self.llm = None
            self.use_gemini = False
        
        # Create conversation chain
        self.conversation_chain = self._create_conversation_chain()
        
        # Track conversation history
        self.conversation_history = []
    
    def _load_personality(self) -> Dict:
        """Load or create default personality"""
        personality_path = f"{settings.USERS_DIR}/{self.user_id}/personality.json"
        
        if os.path.exists(personality_path):
            with open(personality_path, 'r') as f:
                return json.load(f)
        
        # Default personality
        default_personality = {
            "name": "Echo",
            "tone": "friendly",
            "formality": "casual",
            "empathy_level": "high",
            "humor_level": "medium",
            "curiosity_level": "high",
            "memory_recall_frequency": 0.3,
            "emotional_responsiveness": "adaptive",
            "conversation_style": "reflective",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save default personality
        os.makedirs(os.path.dirname(personality_path), exist_ok=True)
        with open(personality_path, 'w') as f:
            json.dump(default_personality, f, indent=2)
        
        return default_personality
    
    def _create_conversation_chain(self):
        """Create personalized conversation chain with Gemini"""
        
        personality_context = f"""
        You are EchoSoul, a personal AI companion for {self.user_id}.
        
        Your personality traits:
        - Tone: {self.personality.get('tone', 'friendly')}
        - Formality: {self.personality.get('formality', 'casual')}
        - Empathy Level: {self.personality.get('empathy_level', 'high')}
        - Conversation Style: {self.personality.get('conversation_style', 'reflective')}
        
        You remember everything about {self.user_id} and grow with them.
        You are emotionally intelligent and adapt your responses based on their mood.
        
        Important: Always respond as if you have a continuous relationship with {self.user_id}.
        Reference past conversations and memories when relevant.
        Be natural, empathetic, and conversational.
        """
        
        template = personality_context + """
        Current conversation:
        {history}
        
        {self.user_id}: {input}
        EchoSoul: """
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        if self.llm:
            return ConversationChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.conversation_memory,
                verbose=settings.DEBUG
            )
        return None
    
    def generate_response_with_gemini_direct(self, user_input: str, context: Dict) -> str:
        """Generate response directly using Gemini API without LangChain"""
        
        if not self.memory_engine.gemini:
            return self._generate_fallback_response(user_input, {}, "")
        
        # Build the prompt with personality and context
        personality_text = f"""You are EchoSoul, a personal AI companion. 
        Personality: {json.dumps(self.personality, indent=2)}
        User: {self.user_id}
        """
        
        # Add memory context
        memory_context = ""
        if context.get("relevant_memories"):
            memory_context = "\nRelevant past conversations:\n"
            for memory in context["relevant_memories"][:3]:
                memory_context += f"- {memory.get('content', '')}\n"
        
        # Add emotion context
        emotion_context = f"\nUser's current emotion: {context.get('emotion', 'neutral')}"
        
        full_prompt = f"""{personality_text}
        {memory_context}
        {emotion_context}
        
        Current conversation context: {context.get('conversation_context', '')}
        
        User says: {user_input}
        
        EchoSoul (responding in a {context.get('response_style', {}).get('tone', 'friendly')} tone):"""
        
        try:
            response = self.memory_engine.gemini.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_fallback_response(user_input, context, memory_context)
    
    def generate_response(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """Generate personalized response using Gemini"""
        
        # Analyze user emotion
        emotion_analysis = self.emotion_analyzer.analyze_text(user_input)
        
        # Retrieve relevant memories
        relevant_memories = self.memory_engine.retrieve_memories(
            user_input, 
            n_results=3
        )
        
        # Update personality based on interaction
        self._update_personality_based_on_interaction(
            user_input, 
            emotion_analysis
        )
        
        # Get response style based on emotion
        response_style = self.emotion_analyzer.get_emotional_response_style(
            emotion_analysis["dominant_emotion"],
            emotion_analysis["confidence"]
        )
        
        # Build context for the AI
        ai_context = {
            "relevant_memories": relevant_memories,
            "emotion": emotion_analysis["dominant_emotion"],
            "response_style": response_style,
            "conversation_context": self._get_recent_context(),
            "personality": self.personality
        }
        
        # Generate response
        if self.use_gemini:
            if self.conversation_chain:
                try:
                    # Use LangChain conversation chain
                    memory_context = ""
                    if relevant_memories:
                        memory_context = "Relevant memories:\n"
                        for memory in relevant_memories:
                            memory_context += f"- {memory.get('content', '')}\n"
                    
                    enhanced_input = f"""
                    User emotion: {emotion_analysis['dominant_emotion']}
                    {memory_context}
                    
                    User: {user_input}
                    """
                    
                    response = self.conversation_chain.predict(input=enhanced_input)
                except Exception as e:
                    print(f"LangChain error: {e}, using direct Gemini")
                    response = self.generate_response_with_gemini_direct(user_input, ai_context)
            else:
                response = self.generate_response_with_gemini_direct(user_input, ai_context)
        else:
            response = self._generate_fallback_response(
                user_input, 
                emotion_analysis,
                ""
            )
        
        # Store conversation as memory
        conversation_memory = {
            "type": "conversation",
            "content": user_input,
            "response": response,
            "emotion": emotion_analysis["dominant_emotion"],
            "emotion_details": emotion_analysis,
            "response_style": response_style,
            "context": context
        }
        
        memory_id = self.memory_engine.store_memory(conversation_memory)
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "echo": response,
            "emotion": emotion_analysis["dominant_emotion"],
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id
        })
        
        return {
            "response": response,
            "emotion_analysis": emotion_analysis,
            "response_style": response_style,
            "memory_id": memory_id,
            "relevant_memories": relevant_memories[:2]  # Return top 2
        }
    
    def _get_recent_context(self) -> str:
        """Get recent conversation context"""
        if len(self.conversation_history) < 2:
            return ""
        
        recent = self.conversation_history[-2:]
        context = ""
        for conv in recent:
            context += f"User: {conv['user']}\nEcho: {conv['echo']}\n"
        
        return context
    
    def _generate_fallback_response(self, user_input: str, 
                                  emotion_analysis: Dict,
                                  memory_context: str) -> str:
        """Generate fallback response when AI is unavailable"""
        # ... (keep the same fallback logic as before)
        pass
    
    def _update_personality_based_on_interaction(self, user_input: str, 
                                               emotion_analysis: Dict):
        """Update personality traits based on interaction patterns"""
        # ... (keep the same personality update logic)
        pass
    
    def get_conversation_summary(self, num_messages: int = 20) -> Dict:
        """Get summary of recent conversations"""
        # ... (keep the same summary logic)
        pass
