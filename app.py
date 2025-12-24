import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from streamlit_chat import message

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
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'echo_ai' not in st.session_state:
        st.session_state.echo_ai = None
    if 'memory_engine' not in st.session_state:
        st.session_state.memory_engine = None
    if 'timeline_manager' not in st.session_state:
        st.session_state.timeline_manager = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'vault_password' not in st.session_state:
        st.session_state.vault_password = None

# Login/Registration Page
def login_page():
    st.title("🌌 Welcome to EchoSoul")
    st.markdown("### Your Personal AI Companion That Remembers and Grows With You")
    
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
    st.session_state.memory_engine = MemoryEngine(user_id)
    st.session_state.echo_ai = EchoSoulAI(user_id)
    st.session_state.timeline_manager = TimelineManager(st.session_state.memory_engine)

# Dashboard Page
def dashboard_page():
    # Sidebar
    with st.sidebar:
        st.title(f"💭 EchoSoul")
        st.markdown(f"**Welcome back!**")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["Chat", "Timeline", "Memory Vault", "Personality", "Settings"],
            key="nav"
        )
        
        # Quick stats
        if st.session_state.memory_engine:
            memories = st.session_state.memory_engine.get_timeline()
            st.markdown("---")
            st.markdown("### Quick Stats")
            st.metric("Total Memories", len(memories))
            
            if memories:
                emotions = [m.get("emotion", "neutral") for m in memories]
                from collections import Counter
                emotion_counts = Counter(emotions)
                if emotion_counts:
                    dominant = max(emotion_counts.items(), key=lambda x: x[1])
                    st.metric("Dominant Emotion", f"{emotion_to_emoji(dominant[0])} {dominant[0]}")
        
        st.markdown("---")
        if st.button("Logout", type="secondary"):
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
    st.title(f"{get_greeting_based_on_time()}! Let's talk 💬")
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(
                    msg["content"],
                    is_user=True,
                    key=f"user_{i}",
                    avatar_style="identicon",
                    seed=st.session_state.user_id
                )
            else:
                # Display emotion badge for Echo responses
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
                    with st.expander("Related Memories"):
                        for memory in msg["memory_references"]:
                            st.caption(f"📅 {format_timestamp(memory['timestamp'], '%b %d')}: {memory['content'][:100]}...")
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Type your message...",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Share your thoughts, feelings, or memories..."
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Voice input placeholder
    with st.expander("🎤 Voice Input (Coming Soon)"):
        st.info("Voice emotion detection and speech-to-text will be available in the next update.")
    
    # Process user input
    if send_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get Echo response
        with st.spinner("Echo is thinking..."):
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
        
        st.rerun()
    
    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💭 Recall a memory", use_container_width=True):
            st.info("What would you like me to remember?")
    
    with col2:
        if st.button("😊 Share a happy moment", use_container_width=True):
            st.info("I'd love to hear about what made you happy!")
    
    with col3:
        if st.button("📝 Journal entry", use_container_width=True):
            st.info("This will be saved in your Memory Vault")

# Timeline Page
def timeline_page():
    st.title("📅 Your Life Timeline")
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("To", datetime.now())
    
    # Get timeline data
    timeline_data = st.session_state.timeline_manager.get_timeline_data(
        start_date.isoformat(),
        end_date.isoformat()
    )
    
    if not timeline_data:
        st.info("No memories found for this period. Start chatting with Echo to create memories!")
        return
    
    # Display emotion timeline chart
    st.subheader("Emotional Journey")
    emotion_chart = st.session_state.timeline_manager.create_emotion_timeline_chart(timeline_data)
    if emotion_chart:
        st.plotly_chart(emotion_chart, use_container_width=True)
    
    # Statistics
    st.subheader("Emotion Statistics")
    stats = st.session_state.timeline_manager.get_emotion_statistics(timeline_data)
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Memories", stats["total_memories"])
        
        with col2:
            dominant = stats.get("dominant_emotion", "neutral")
            st.metric("Dominant Emotion", f"{emotion_to_emoji(dominant)} {dominant}")
        
        with col3:
            diversity = stats.get("emotional_diversity", 0)
            st.metric("Emotional Diversity", f"{diversity*100:.1f}%")
        
        # Emotion distribution
        st.subheader("Emotion Distribution")
        emotion_df = pd.DataFrame(
            list(stats["emotion_distribution"].items()),
            columns=["Emotion", "Count"]
        )
        
        if not emotion_df.empty:
            fig = go.Figure(data=[
                go.Pie(
                    labels=emotion_df["Emotion"],
                    values=emotion_df["Count"],
                    hole=.3,
                    marker=dict(colors=[emotion_to_color(e) for e in emotion_df["Emotion"]])
                )
            ])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("Insights")
        for insight in stats.get("insights", []):
            st.info(f"💡 {insight}")
    
    # Timeline view
    st.subheader("Memory Timeline")
    
    for memory in timeline_data[-20:]:  # Show last 20 memories
        emotion = memory.get("emotion", "neutral")
        emotion_color = emotion_to_color(emotion)
        
        with st.expander(
            f"{emotion_to_emoji(emotion)} {format_timestamp(memory['timestamp'], '%b %d, %Y %I:%M %p')} - {memory['type'].title()}",
            expanded=False
        ):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">{emotion_to_emoji(emotion)}</div>
                    <div style="color: {emotion_color}; font-weight: bold;">{emotion.title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write(memory.get("full_content", ""))
                
                if memory.get("response_style"):
                    st.caption(f"**Response Style:** {memory['response_style'].get('tone', 'neutral').title()}")
                
                # Delete button
                if st.button("Delete", key=f"delete_{memory['id']}"):
                    # Implement delete functionality
                    st.warning("Delete functionality will be implemented in the next version")

# Memory Vault Page
def vault_page():
    st.title("🔒 Private Memory Vault")
    st.markdown("Your encrypted, private memories. Only you can access these.")
    
    # Password protection
    if 'vault_unlocked' not in st.session_state:
        st.session_state.vault_unlocked = False
    
    if not st.session_state.vault_unlocked:
        st.markdown("### Enter Vault Password")
        vault_password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Unlock Vault", type="primary", use_container_width=True):
                # Simple password check (in production, use proper authentication)
                if vault_password == "echosoul2024":  # Default for demo
                    st.session_state.vault_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        
        with col2:
            if st.button("Set New Password", type="secondary", use_container_width=True):
                st.info("Password reset feature coming soon")
        
        return
    
    # Vault unlocked
    st.success("🔓 Vault Unlocked")
    
    # Get vault memories
    vault_memories = st.session_state.memory_engine.get_vault_memories()
    
    # Add new vault memory
    st.subheader("Add Private Memory")
    
    with st.form("add_vault_memory"):
        memory_title = st.text_input("Title")
        memory_content = st.text_area("Content", height=150)
        memory_tags = st.multiselect(
            "Tags",
            ["Personal", "Secret", "Dream", "Fear", "Goal", "Memory", "Confession"],
            default=["Personal"]
        )
        memory_emotion = st.selectbox(
            "Emotion",
            ["neutral", "joy", "sadness", "love", "fear", "anger", "anxiety", "excitement"]
        )
        
        submitted = st.form_submit_button("Encrypt & Save to Vault", type="primary")
        
        if submitted and memory_content:
            vault_memory = {
                "title": memory_title,
                "content": memory_content,
                "tags": memory_tags,
                "emotion": memory_emotion,
                "type": "vault",
                "encrypted": True
            }
            
            memory_id = st.session_state.memory_engine.store_memory(vault_memory, is_vault=True)
            st.success(f"Memory encrypted and saved! ID: {memory_id}")
            st.rerun()
    
    # Display vault memories
    st.subheader("Your Private Memories")
    
    if not vault_memories:
        st.info("No private memories yet. Add one above!")
        return
    
    for i, memory in enumerate(vault_memories):
        with st.expander(
            f"🔒 {memory.get('title', 'Untitled Memory')} - {format_timestamp(memory.get('timestamp', ''), '%b %d, %Y')}",
            expanded=False
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(memory.get("content", ""))
                
                # Display tags
                tags = memory.get("tags", [])
                if tags:
                    tag_html = " ".join([f"<span style='background-color: #e0e0e0; padding: 4px 8px; border-radius: 12px; margin: 2px; font-size: 0.8rem;'>{tag}</span>" for tag in tags])
                    st.markdown(tag_html, unsafe_allow_html=True)
            
            with col2:
                emotion = memory.get("emotion", "neutral")
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">{emotion_to_emoji(emotion)}</div>
                    <div style="color: {emotion_to_color(emotion)}; font-weight: bold;">{emotion.title()}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Delete Forever", key=f"delete_vault_{i}", type="secondary"):
                    st.warning("Permanent deletion coming in next version")
    
    # Lock vault button
    st.markdown("---")
    if st.button("🔒 Lock Vault", type="secondary", use_container_width=True):
        st.session_state.vault_unlocked = False
        st.rerun()

# Personality Page
def personality_page():
    st.title("🎭 Your EchoSoul Personality")
    st.markdown("Watch how your AI companion evolves with you.")
    
    # Get personality data
    personality = st.session_state.echo_ai.personality if st.session_state.echo_ai else {}
    
    if not personality:
        st.info("Personality data not available yet.")
        return
    
    # Personality traits
    st.subheader("Current Personality Traits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Tone", per
