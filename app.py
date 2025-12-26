
import streamlit as st
import json
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from streamlit_chat import message
import google.genai as genai  # New package
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load custom modules
try:
    from config import settings
    from memory_engine import MemoryEngine
    from emotion_analyzer import EmotionAnalyzer
    from ai_brain import EchoSoulAI
    from timeline_manager import TimelineManager
    from utils import (
        format_timestamp, emotion_to_emoji, emotion_to_color,
        format_memory_for_display, generate_user_id, validate_email,
        calculate_sentiment_score, create_progress_bar, get_greeting_based_on_time
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all required modules are in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="EchoSoul - Your AI Companion",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6A6A6A;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    .emotion-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 2px;
    }
    .memory-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid;
        background-color: #f8f9fa;
    }
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        margin: 5px 0;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #007AFF;
        color: white;
        margin-left: auto;
    }
    .echo-bubble {
        background-color: #E8E8E8;
        color: #333;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    .timeline-event {
        padding: 10px;
        margin: 10px 0;
        border-left: 3px solid;
        background-color: #f5f5f5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2;
        color: white;
    }
    .api-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-good {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .personality-badge {
        display: inline-block;
        padding: 6px 12px;
        margin: 3px;
        border-radius: 20px;
        background-color: #e3f2fd;
        color: #1565c0;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Check for API keys
def check_api_keys():
    """Check if required API keys are available"""
    missing_keys = []
    
    # Check Google API Key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        # Check in settings
        if not hasattr(settings, 'GOOGLE_API_KEY') or not settings.GOOGLE_API_KEY:
            missing_keys.append("Google Gemini API Key")
    
    if missing_keys:
        st.warning(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
        
        with st.expander("How to get API keys", expanded=True):
            st.markdown("""
            ### 1. Google AI Studio (Gemini) API Key
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Get API Key" or "Create API Key"
            4. Copy the API key
            
            ### 2. Add to your `.env` file:
            ```env
            GOOGLE_API_KEY=your_google_api_key_here
            ENCRYPTION_KEY=your-secure-encryption-key-here
            ```
            
            ### 3. Or set as environment variable:
            ```bash
            export GOOGLE_API_KEY="your_key_here"
            ```
            """)
        
        # Option to input API key directly
        with st.form("api_key_form"):
            google_key = st.text_input("Enter Google Gemini API Key", type="password")
            
            if st.form_submit_button("Use This Key"):
                if google_key:
                    os.environ["GOOGLE_API_KEY"] = google_key
                    settings.GOOGLE_API_KEY = google_key
                    st.success("API key saved for this session!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")
        
        # Option to continue without API key (with limited functionality)
        if st.button("Continue with Limited Features"):
            st.session_state.continue_without_key = True
            st.rerun()
        
        return False
    
    return True

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    default_states = {
        'user_id': None,
        'echo_ai': None,
        'memory_engine': None,
        'timeline_manager': None,
        'conversation_history': [],
        'current_page': "login",
        'user_email': None,
        'vault_password': None,
        'vault_unlocked': False,
        'messages': [],
        'continue_without_key': False,
        'api_checked': False,
        'personality_traits': {},
        'theme': 'light'
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize Google Gemini
def init_gemini():
    """Initialize Google Gemini API"""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, 'GOOGLE_API_KEY', None)
        if google_api_key:
            genai.configure(api_key=google_api_key)
            
            # Test the API
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Hello", safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            })
            
            st.session_state.gemini_available = True
            return True
        else:
            st.session_state.gemini_available = False
            return False
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        st.session_state.gemini_available = False
        return False

# Login/Registration Page
def login_page():
    """Login and registration page"""
    st.title("🌌 Welcome to EchoSoul")
    st.markdown("### Your Personal AI Companion That Remembers and Grows With You")
    
    # Show API status
    if hasattr(st.session_state, 'gemini_available'):
        if st.session_state.gemini_available:
            st.success("✅ Google Gemini API is connected!")
        else:
            st.warning("⚠️ Google Gemini API is not available. Using fallback mode.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if validate_email(email) and password:
                user_id = generate_user_id(email)
                st.session_state.user_id = user_id
                st.session_state.user_email = email
                st.session_state.current_page = "dashboard"
                initialize_user_components(user_id)
                st.rerun()
            else:
                st.error("Please enter valid email and password")
    
    with col2:
        st.subheader("Register")
        new_email = st.text_input("Email", key="register_email")
        new_name = st.text_input("Name", key="register_name")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm")
        
        if st.button("Create Account", type="secondary", use_container_width=True):
            if not validate_email(new_email):
                st.error("Please enter a valid email")
            elif new_password != confirm_password:
                st.error("Passwords don't match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                user_id = generate_user_id(new_email)
                
                # Create user directory
                user_dir = f"{settings.USERS_DIR}/{user_id}"
                os.makedirs(user_dir, exist_ok=True)
                
                # Save user profile
                profile = {
                    "email": new_email,
                    "name": new_name,
                    "created_at": datetime.now().isoformat(),
                    "last_login": datetime.now().isoformat()
                }
                
                with open(f"{user_dir}/profile.json", "w") as f:
                    json.dump(profile, f, indent=2)
                
                st.session_state.user_id = user_id
                st.session_state.user_email = new_email
                st.session_state.current_page = "dashboard"
                initialize_user_components(user_id)
                st.success("Account created successfully!")
                st.rerun()

# Initialize user components
def initialize_user_components(user_id: str):
    """Initialize all components for a user"""
    try:
        st.session_state.memory_engine = MemoryEngine(user_id)
        st.session_state.echo_ai = EchoSoulAI(user_id)
        st.session_state.timeline_manager = TimelineManager(st.session_state.memory_engine)
        
        # Load personality traits
        personality_path = f"{settings.USERS_DIR}/{user_id}/personality.json"
        if os.path.exists(personality_path):
            with open(personality_path, 'r') as f:
                st.session_state.personality_traits = json.load(f)
    except Exception as e:
        st.error(f"Error initializing user components: {str(e)}")
        st.info("Some features may not work correctly.")

# Dashboard Page
def dashboard_page():
    """Main dashboard page"""
    # Sidebar
    with st.sidebar:
        st.title(f"💭 EchoSoul")
        
        # User info
        if st.session_state.user_email:
            st.markdown(f"**Welcome back!**")
            st.caption(f"User: {st.session_state.user_email}")
        
        # Navigation using tabs
        page = st.radio(
            "Navigate",
            ["Chat", "Timeline", "Memory Vault", "Personality", "Settings"],
            key="nav",
            label_visibility="collapsed"
        )
        
        # Quick stats
        if st.session_state.memory_engine:
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            
            try:
                memories = st.session_state.memory_engine.get_timeline()
                if memories:
                    st.metric("Total Memories", len(memories))
                    
                    # Get emotion distribution
                    emotions = [m.get("emotion", "neutral") for m in memories]
                    from collections import Counter
                    emotion_counts = Counter(emotions)
                    if emotion_counts:
                        dominant = max(emotion_counts.items(), key=lambda x: x[1])
                        st.metric("Dominant Emotion", f"{emotion_to_emoji(dominant[0])} {dominant[0]}")
                else:
                    st.info("No memories yet. Start chatting!")
            except:
                pass
        
        # API Status
        st.markdown("---")
        st.markdown("### 🔧 Status")
        if st.session_state.get('gemini_available'):
            st.success("Gemini: Connected")
        else:
            st.warning("Gemini: Not Connected")
        
        st.markdown("---")
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.session_state.current_page = "login"
            st.rerun()
    
    # Main content based on selection
    if page == "Chat":
        chat_page()
    elif page == "Timeline":
        timeline_page()
    elif page == "Memory Vault":
        vault_page()
    elif page == "Personality":
        personality_page()
    elif page == "Settings":
        settings_page()

# Chat Page
def chat_page():
    """Main chat interface"""
    st.title(f"{get_greeting_based_on_time()}! Let's talk 💬")
    
    # Show Gemini status
    if not st.session_state.get('gemini_available'):
        st.warning("⚠️ Google Gemini is not connected. Using fallback responses.")
        st.info("To enable full AI capabilities, add your Google API key in Settings.")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container(height=500)
    
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                # User message
                message(
                    msg["content"],
                    is_user=True,
                    key=f"user_{i}",
                    avatar_style="personas",
                    seed=st.session_state.user_id
                )
            else:
                # EchoSoul message with emotion
                emotion = msg.get("emotion", "neutral")
                emotion_color = emotion_to_color(emotion)
                
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.markdown(f"<div style='color: {emotion_color}; font-size: 24px;'>{emotion_to_emoji(emotion)}</div>", 
                               unsafe_allow_html=True)
                with col2:
                    message(
                        msg["content"],
                        is_user=False,
                        key=f"echo_{i}",
                        avatar_style="bottts",
                        seed="EchoSoul"
                    )
                
                # Show memory references if available
                if msg.get("memory_references"):
                    with st.expander("💭 Related Memories", expanded=False):
                        for memory in msg["memory_references"][:3]:
                            timestamp = format_timestamp(memory.get('timestamp', ''), '%b %d, %Y')
                            preview = memory.get('content', '')[:100]
                            st.caption(f"📅 **{timestamp}**: {preview}...")
    
    # Chat input area
    st.markdown("---")
    
    # Input options
    input_col1, input_col2, input_col3 = st.columns([1, 4, 1])
    
    with input_col1:
        input_mode = st.radio("Input", ["Text", "Voice"], horizontal=True, label_visibility="collapsed")
    
    with input_col2:
        if input_mode == "Text":
            user_input = st.text_input(
                "Type your message...",
                key="chat_input",
                label_visibility="collapsed",
                placeholder="Share your thoughts, feelings, or memories..."
            )
        else:
            st.info("🎤 Voice input coming soon!")
            user_input = ""
    
    with input_col3:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Quick action buttons
    st.markdown("### Quick Actions")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("💭 Recall Memory", use_container_width=True, help="Ask about something you mentioned before"):
            st.session_state.quick_action = "recall"
    
    with quick_col2:
        if st.button("🎯 Daily Check-in", use_container_width=True, help="How are you feeling today?"):
            st.session_state.quick_action = "checkin"
    
    with quick_col3:
        if st.button("📖 Life Story", use_container_width=True, help="Share an important life event"):
            st.session_state.quick_action = "story"
    
    # Handle quick actions
    if 'quick_action' in st.session_state:
        quick_action = st.session_state.pop('quick_action')
        if quick_action == "recall":
            user_input = "Can you remember something I told you before?"
        elif quick_action == "checkin":
            user_input = "Let's do a daily check-in. How am I doing emotionally?"
        elif quick_action == "story":
            user_input = "I want to share an important story from my life."
        
        # Auto-fill the input
        if 'chat_input' in st.session_state:
            st.session_state.chat_input = user_input
        st.rerun()
    
    # Process user input
    if send_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show typing indicator
        with st.spinner("💭 Echo is thinking..."):
            try:
                # Get Echo response
                if st.session_state.echo_ai:
                    response = st.session_state.echo_ai.generate_response(user_input)
                    
                    # Add Echo response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["response"],
                        "emotion": response["emotion_analysis"]["dominant_emotion"],
                        "memory_references": response.get("relevant_memories", [])
                    })
                    
                    # Update conversation history
                    st.session_state.conversation_history.append({
                        "user": user_input,
                        "echo": response["response"],
                        "timestamp": datetime.now().isoformat(),
                        "emotion": response["emotion_analysis"]["dominant_emotion"]
                    })
                else:
                    # Fallback response
                    fallback_response = "I'm here to listen. Tell me more about how you're feeling."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback_response,
                        "emotion": "neutral"
                    })
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I encountered an error. Please try again.",
                    "emotion": "neutral"
                })
        
        st.rerun()

# Timeline Page
def timeline_page():
    """Timeline visualization page"""
    st.title("📅 Your Life Timeline")
    
    if not st.session_state.memory_engine:
        st.info("Please start chatting first to create memories!")
        return
    
    # Date range filter
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_date = st.date_input("From", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("To", datetime.now())
    with col3:
        filter_type = st.selectbox("Filter", ["All", "Conversations", "Memories", "Events"])
    
    # Get timeline data
    try:
        timeline_data = st.session_state.timeline_manager.get_timeline_data(
            start_date.isoformat(),
            end_date.isoformat()
        )
        
        # Apply type filter
        if filter_type != "All":
            if filter_type == "Conversations":
                timeline_data = [m for m in timeline_data if m.get("type") == "conversation"]
            elif filter_type == "Memories":
                timeline_data = [m for m in timeline_data if m.get("type") == "memory"]
            elif filter_type == "Events":
                timeline_data = [m for m in timeline_data if m.get("type") == "event"]
        
        if not timeline_data:
            st.info("No memories found for this period. Start chatting with Echo to create memories!")
            
            # Show suggestions
            with st.expander("How to create memories"):
                st.markdown("""
                - **Share your feelings**: "Today I felt..."
                - **Talk about events**: "Yesterday I went to..."
                - **Discuss memories**: "I remember when..."
                - **Ask questions**: "What do you think about..."
                """)
            return
        
        # Display emotion timeline chart
        st.subheader("🎭 Emotional Journey")
        emotion_chart = st.session_state.timeline_manager.create_emotion_timeline_chart(timeline_data)
        if emotion_chart:
            st.plotly_chart(emotion_chart, use_container_width=True)
        else:
            st.info("Not enough data for timeline visualization yet.")
        
        # Statistics section
        st.subheader("📊 Emotion Statistics")
        stats = st.session_state.timeline_manager.get_emotion_statistics(timeline_data)
        
        if stats:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Memories", stats["total_memories"])
            
            with col2:
                dominant = stats.get("dominant_emotion", "neutral")
                st.metric("Dominant Emotion", f"{emotion_to_emoji(dominant)}")
            
            with col3:
                diversity = stats.get("emotional_diversity", 0)
                st.metric("Emotional Diversity", f"{diversity*100:.0f}%")
            
            with col4:
                unique = len(stats.get("emotion_distribution", {}))
                st.metric("Unique Emotions", unique)
            
            # Emotion distribution chart
            if stats.get("emotion_distribution"):
                st.subheader("Emotion Distribution")
                emotion_df = pd.DataFrame(
                    list(stats["emotion_distribution"].items()),
                    columns=["Emotion", "Count"]
                )
                
                # Create pie chart
                fig = go.Figure(data=[
                    go.Pie(
                        labels=emotion_df["Emotion"],
                        values=emotion_df["Count"],
                        hole=.3,
                        marker=dict(colors=[emotion_to_color(e) for e in emotion_df["Emotion"]]),
                        textinfo='label+percent'
                    )
                ])
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            st.subheader("💡 Insights")
            insights = stats.get("insights", [])
            if insights:
                for insight in insights:
                    st.info(f"• {insight}")
            else:
                st.info("Keep chatting to generate insights about your emotional patterns!")
        
        # Timeline view
        st.subheader("📝 Memory Timeline")
        
        # Group by date
        memories_by_date = {}
        for memory in timeline_data[-50:]:  # Show last 50 memories
            date_str = memory.get("date", "Unknown")
            if date_str not in memories_by_date:
                memories_by_date[date_str] = []
            memories_by_date[date_str].append(memory)
        
        # Display by date
        for date_str in sorted(memories_by_date.keys(), reverse=True):
            with st.expander(f"📅 {date_str} ({len(memories_by_date[date_str])} memories)", expanded=False):
                for memory in memories_by_date[date_str]:
                    emotion = memory.get("emotion", "neutral")
                    emotion_color = emotion_to_color(emotion)
                    
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 2rem;">{emotion_to_emoji(emotion)}</div>
                            <div style="color: {emotion_color}; font-weight: bold;">{emotion.title()}</div>
                            <div style="font-size: 0.8rem; color: #666;">
                                {memory.get('type', 'memory').title()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.write(memory.get("full_content", ""))
                        
                        # Memory actions
                        col_action1, col_action2 = st.columns(2)
                        with col_action1:
                            if st.button("🔍 View Details", key=f"view_{memory['id']}", use_container_width=True):
                                st.session_state.selected_memory = memory
                                st.rerun()
                        with col_action2:
                            if st.button("🗑️ Delete", key=f"delete_{memory['id']}", use_container_width=True, type="secondary"):
                                if st.checkbox(f"Confirm deletion of memory from {date_str}"):
                                    # Implement delete functionality
                                    st.warning("Delete functionality coming in next version")
        
    except Exception as e:
        st.error(f"Error loading timeline: {str(e)}")
        st.info("Try chatting with Echo first to create some memories!")

# Memory Vault Page
def vault_page():
    """Encrypted memory vault page"""
    st.title("🔒 Private Memory Vault")
    st.markdown("Your encrypted, private memories. Only you can access these.")
    
    # Password protection
    if 'vault_unlocked' not in st.session_state:
        st.session_state.vault_unlocked = False
    
    if not st.session_state.vault_unlocked:
        # Vault lock screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Enter Vault Password")
            vault_password = st.text_input("Password", type="password", key="vault_password_input")
            
            if st.button("🔓 Unlock Vault", type="primary", use_container_width=True):
                # Simple password check (in production, use proper authentication)
                if vault_password == "echosoul":  # Default for demo
                    st.session_state.vault_unlocked = True
                    st.success("Vault unlocked!")
                    st.rerun()
                else:
                    st.error("Incorrect password")
            
            st.markdown("---")
            if st.button("🆕 Set New Password", type="secondary", use_container_width=True):
                st.info("Password reset feature coming soon")
        
        return
    
    # Vault unlocked - show content
    st.success("🔓 Vault Unlocked")
    
    # Get vault memories
    try:
        vault_memories = st.session_state.memory_engine.get_vault_memories()
        
        # Add new vault memory
        st.subheader("➕ Add Private Memory")
        
        with st.form("add_vault_memory"):
            col1, col2 = st.columns(2)
            
            with col1:
                memory_title = st.text_input("Title", placeholder="Memory title")
                memory_emotion = st.selectbox(
                    "Primary Emotion",
                    ["neutral", "joy", "sadness", "love", "fear", "anger", "anxiety", "excitement", "contentment"]
                )
            
            with col2:
                memory_type = st.selectbox(
                    "Memory Type",
                    ["personal", "secret", "dream", "goal", "reflection", "confession"]
                )
                memory_tags = st.multiselect(
                    "Tags",
                    ["Personal", "Secret", "Dream", "Fear", "Goal", "Memory", "Confession", "Important", "Healing"],
                    default=["Personal"]
                )
            
            memory_content = st.text_area(
                "Content", 
                height=150,
                placeholder="Write your private memory here...",
                help="This will be encrypted and stored securely"
            )
            
            submitted = st.form_submit_button("🔒 Encrypt & Save to Vault", type="primary")
            
            if submitted:
                if memory_content:
                    vault_memory = {
                        "title": memory_title or "Untitled Memory",
                        "content": memory_content,
                        "tags": memory_tags,
                        "emotion": memory_emotion,
                        "type": memory_type,
                        "encrypted": True
                    }
                    
                    memory_id = st.session_state.memory_engine.store_memory(vault_memory, is_vault=True)
                    st.success(f"✅ Memory encrypted and saved!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Please enter memory content")
        
        # Display vault memories
        st.subheader(f"📁 Your Private Memories ({len(vault_memories)})")
        
        if not vault_memories:
            st.info("No private memories yet. Add one above!")
        else:
            # Filter and search
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_query = st.text_input("Search memories...", placeholder="Type to search")
            with search_col2:
                sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Emotion"])
            
            # Filter memories
            filtered_memories = vault_memories
            if search_query:
                filtered_memories = [
                    m for m in vault_memories 
                    if search_query.lower() in m.get('content', '').lower() 
                    or search_query.lower() in m.get('title', '').lower()
                ]
            
            # Sort memories
            if sort_by == "Newest":
                filtered_memories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            elif sort_by == "Oldest":
                filtered_memories.sort(key=lambda x: x.get('timestamp', ''))
            elif sort_by == "Emotion":
                filtered_memories.sort(key=lambda x: x.get('emotion', ''))
            
            # Display memories
            for i, memory in enumerate(filtered_memories):
                emotion = memory.get("emotion", "neutral")
                emotion_color = emotion_to_color(emotion)
                
                with st.expander(
                    f"{emotion_to_emoji(emotion)} {memory.get('title', 'Untitled Memory')} - {format_timestamp(memory.get('timestamp', ''), '%b %d, %Y')}",
                    expanded=False
                ):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(memory.get("content", ""))
                        
                        # Display tags
                        tags = memory.get("tags", [])
                        if tags:
                            tag_html = " ".join([f"<span class='personality-badge'>{tag}</span>" for tag in tags])
                            st.markdown(tag_html, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 2rem;">{emotion_to_emoji(emotion)}</div>
                            <div style="color: {emotion_color}; font-weight: bold;">{emotion.title()}</div>
                            <div style="font-size: 0.8rem; color: #666; margin-top: 10px;">
                                {memory.get('type', 'personal').title()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Memory actions
                        if st.button("🗑️ Delete", key=f"delete_vault_{i}", use_container_width=True):
                            if st.checkbox("Confirm permanent deletion", key=f"confirm_delete_{i}"):
                                st.warning("Permanent deletion coming in next version")
        
        # Lock vault button
        st.markdown("---")
        if st.button("🔒 Lock Vault", type="secondary", use_container_width=True):
            st.session_state.vault_unlocked = False
            st.success("Vault locked!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error accessing vault: {str(e)}")
        st.info("Try unlocking the vault again.")

# Personality Page
def personality_page():
    """Personality customization and analysis page"""
    st.title("🎭 Your EchoSoul Personality")
    st.markdown("Watch how your AI companion evolves with you.")
    
    if not st.session_state.echo_ai:
        st.info("Please start chatting first to develop personality!")
        return
    
    # Get personality data
    try:
        personality = st.session_state.echo_ai.personality
        conversation_summary = st.session_state.echo_ai.get_conversation_summary()
        
        if not personality:
            st.info("Personality data not available yet. Start chatting to develop your EchoSoul's personality!")
            return
        
        # Personality overview
        st.subheader("📋 Current Personality Traits")
        
        # Display personality traits as badges
        traits_col1, traits_col2 = st.columns(2)
        
        with traits_col1:
            st.markdown("**Core Traits:**")
            core_traits = {
                "Tone": personality.get("tone", "friendly"),
                "Formality": personality.get("formality", "casual"),
                "Empathy Level": personality.get("empathy_level", "high")
            }
            
            for trait, value in core_traits.items():
                st.markdown(f"**{trait}:** {value.title()}")
                st.progress(
                    {"low": 0.25, "medium": 0.5, "high": 0.75, "very_high": 1.0}.get(
                        value.lower(), 0.5
                    )
                )
        
        with traits_col2:
            st.markdown("**Interaction Style:**")
            style_traits = {
                "Humor Level": personality.get("humor_level", "medium"),
                "Curiosity Level": personality.get("curiosity_level", "high"),
                "Memory Recall": f"{personality.get('memory_recall_frequency', 0.3)*100:.0f}%"
            }
            
            for trait, value in style_traits.items():
                if trait == "Memory Recall":
                    st.markdown(f"**{trait}:** {value}")
                else:
                    st.markdown(f"**{trait}:** {value.title()}")
                    if trait != "Memory Recall":
                        st.progress(
                            {"none": 0, "low": 0.25, "medium": 0.5, "high": 0.75}.get(
                                value.lower(), 0.5
                            )
                        )
        
        # Conversation analysis
        st.subheader("💬 Conversation Analysis")
        
        if conversation_summary:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total = conversation_summary.get("total_conversations", 0)
                st.metric("Total Conversations", total)
            
            with col2:
                trend = conversation_summary.get("emotion_trend", "stable")
                st.metric("Emotion Trend", trend.title())
            
            with col3:
                pattern = conversation_summary.get("dominant_emotion_pattern", "neutral")
                st.metric("Dominant Pattern", pattern.title())
            
            # Recent topics
            topics = conversation_summary.get("recent_topics", [])
            if topics:
                st.markdown("**Recent Topics:**")
                topic_html = " ".join([f"<span class='personality-badge'>{topic.title()}</span>" for topic in topics])
                st.markdown(topic_html, unsafe_allow_html=True)
        
        # Personality evolution
        st.subheader("📈 Personality Evolution")
        
        # Mock evolution chart (in real app, track historical changes)
        evolution_data = pd.DataFrame({
            'Date': pd.date_range(end=datetime.now(), periods=7, freq='D'),
            'Empathy': [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'Curiosity': [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
            'Formality': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        })
        
        fig = go.Figure()
        for column in ['Empathy', 'Curiosity', 'Formality']:
            fig.add_trace(go.Scatter(
                x=evolution_data['Date'],
                y=evolution_data[column],
                name=column,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Personality Evolution Over Time",
            xaxis_title="Date",
            yaxis_title="Level",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Personality customization
        st.subheader("⚙️ Customize Personality")
        
        with st.form("customize_personality"):
            st.markdown("Adjust how EchoSoul interacts with you:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_tone = st.select_slider(
                    "Communication Tone",
                    options=["professional", "serious", "balanced", "friendly", "warm", "playful"],
                    value=personality.get("tone", "friendly")
                )
                
                new_empathy = st.select_slider(
                    "Empathy Level",
                    options=["low", "medium", "high", "very_high"],
                    value=personality.get("empathy_level", "high")
                )
            
            with col2:
                new_humor = st.select_slider(
                    "Humor Level",
                    options=["none", "low", "medium", "high"],
                    value=personality.get("humor_level", "medium")
                )
                
                new_memory_freq = st.slider(
                    "Memory Recall Frequency",
                    min_value=0.0,
                    max_value=1.0,
                    value=personality.get("memory_recall_frequency", 0.3),
                    step=0.1,
                    help="How often EchoSoul should reference past conversations"
                )
            
            # Additional customization
            with st.expander("Advanced Settings"):
                new_formality = st.select_slider(
                    "Formality Level",
                    options=["very_formal", "formal", "balanced", "casual", "very_casual"],
                    value=personality.get("formality", "casual")
                )
                
                new_response_length = st.select_slider(
                    "Response Length",
                    options=["concise", "medium", "detailed"],
                    value=personality.get("response_length", "medium")
                )
            
            if st.form_submit_button("💾 Update Personality", type="primary"):
                # Update personality
                updates = {
                    "tone": new_tone,
                    "empathy_level": new_empathy,
                    "humor_level": new_humor,
                    "memory_recall_frequency": new_memory_freq,
                    "formality": new_formality,
                    "response_length": new_response_length,
                    "last_updated": datetime.now().isoformat()
                }
                
                for trait, value in updates.items():
                    st.session_state.memory_engine.update_personality_trait(trait, value)
                
                st.success("✅ Personality updated! Echo will adapt to these changes.")
                st.balloons()
                st.rerun()
        
        # Personality insights
        st.subheader("💡 Personality Insights")
        
        insights = []
        if personality.get("empathy_level") == "very_high":
            insights.append("Your EchoSoul is highly empathetic, great for emotional support!")
        
        if personality.get("humor_level") in ["medium", "high"]:
            insights.append("EchoSoul incorporates humor in conversations.")
        
        if personality.get("memory_recall_frequency", 0) > 0.5:
            insights.append("EchoSoul frequently references past conversations, showing strong memory recall.")
        
        if insights:
            for insight in insights:
                st.info(f"• {insight}")
        else:
            st.info("Keep chatting to develop more personality insights!")
            
    except Exception as e:
        st.error(f"Error loading personality data: {str(e)}")
        st.info("Try chatting more to develop personality traits.")

# Settings Page
def settings_page():
    """Settings and configuration page"""
    st.title("⚙️ Settings")
    
    # User profile
    st.subheader("👤 Profile Settings")
    
    # Load user profile
    user_dir = f"{settings.USERS_DIR}/{st.session_state.user_id}"
    profile_path = f"{user_dir}/profile.json"
    
    if os.path.exists(profile_path):
        with open(profile_path, 'r') as f:
            profile = json.load(f)
    else:
        profile = {}
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name", value=profile.get("name", ""))
            email = st.text_input("Email", value=profile.get("email", st.session_state.user_email), disabled=True)
        
        with col2:
            birth_date = st.date_input(
                "Birth Date", 
                value=datetime.strptime(profile.get("birth_date", "2000-01-01"), "%Y-%m-%d") if profile.get("birth_date") else datetime(2000, 1, 1)
            )
            timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT", "IST", "CET"], index=0)
        
        bio = st.text_area(
            "Bio", 
            value=profile.get("bio", ""), 
            height=100,
            placeholder="Tell me about yourself..."
        )
        
        submitted = st.form_submit_button("💾 Save Profile", type="primary")
        
        if submitted:
            profile.update({
                "name": name,
                "birth_date": birth_date.isoformat(),
                "timezone": timezone,
                "bio": bio,
                "updated_at": datetime.now().isoformat()
            })
            
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            st.success("✅ Profile updated successfully!")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    with st.form("api_config_form"):
        st.markdown("### Google Gemini API")
        google_key = st.text_input(
            "Google API Key", 
            value=os.getenv("GOOGLE_API_KEY", "") or getattr(settings, 'GOOGLE_API_KEY', ""),
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro"],
            index=0,
            help="Select which Gemini model to use"
        )
        
        if st.form_submit_button("🔧 Update API Configuration", type="primary"):
            if google_key:
                os.environ["GOOGLE_API_KEY"] = google_key
                settings.GOOGLE_API_KEY = google_key
                settings.GEMINI_MODEL = gemini_model
                
                # Reinitialize Gemini
                init_gemini()
                
                st.success("✅ API configuration updated!")
                st.info("You may need to restart the app for changes to take full effect.")
            else:
                st.warning("Please enter a Google API key")
    
    # App settings
    st.subheader("🎨 App Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        notifications = st.checkbox("Enable Notifications", value=True)
        auto_save = st.checkbox("Auto-save Conversations", value=True)
    
    with col2:
        voice_enabled = st.checkbox("Enable Voice Features", value=False)
        data_retention = st.select_slider(
            "Data Retention",
            options=["30 days", "90 days", "1 year", "Forever"],
            value="Forever"
        )
    
    if st.button("💾 Save App Settings", type="primary"):
        st.session_state.theme = theme
        st.success("App settings saved!")
    
    # Data management
    st.subheader("🗃️ Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Export All Data", use_container_width=True, help="Export all your memories and conversations"):
            st.info("Export feature coming soon")
    
    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True, help="Clear current conversation history"):
            if st.checkbox("I understand this cannot be undone", key="clear_confirm"):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.success("Chat history cleared!")
    
    with col3:
        if st.button("🔄 Reset Personality", use_container_width=True, help="Reset EchoSoul's personality to default"):
            if st.checkbox("Confirm personality reset", key="reset_confirm"):
                st.warning("Personality reset feature coming soon")
    
    # Legacy mode
    st.subheader("🌳 Legacy Mode")
    st.markdown("Preserve your EchoSoul for future generations.")
    
    legacy_enabled = st.checkbox("Enable Legacy Mode")
    
    if legacy_enabled:
        legacy_message = st.text_area(
            "Legacy Message", 
            placeholder="What would you like future generations to know about you?",
            height=150
        )
        
        legacy_recipients = st.text_input("Legacy Recipients (comma-separated emails)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🛡️ Setup Legacy", type="primary", use_container_width=True):
                st.success("Legacy mode activated! Your EchoSoul will be preserved.")
        with col2:
            if st.button("📜 Preview Legacy", use_container_width=True):
                st.info("Legacy preview coming soon")
    
    # Danger zone
    st.subheader("⚠️ Danger Zone")
    
    with st.expander("Account Deletion", expanded=False):
        st.warning("This action is permanent and cannot be undone!")
        
        delete_email = st.text_input("Enter your email to confirm deletion")
        
        if st.button("🗑️ Delete Account", type="secondary", disabled=True):
            if delete_email == st.session_state.user_email:
                st.error("Account deletion feature coming soon")
            else:
                st.error("Email does not match")

# Main app logic
def main():
    """Main application entry point"""
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Check API keys (only once per session)
    if not st.session_state.get('api_checked', False):
        if not st.session_state.get('continue_without_key', False):
            if not check_api_keys():
                st.stop()
        st.session_state.api_checked = True
    
    # Initialize Gemini
    if not st.session_state.get('gemini_initialized', False):
        init_gemini()
        st.session_state.gemini_initialized = True
    
    # Page routing
    if st.session_state.current_page == "login":
        login_page()
    elif st.session_state.current_page == "dashboard":
        dashboard_page()
    else:
        st.session_state.current_page = "login"
        st.rerun()

if __name__ == "__main__":
    main()
