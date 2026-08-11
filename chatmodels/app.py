import streamlit as st
import os
 
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
 
load_dotenv()
 
 
# =========================================================
# PAGE CONFIGURATION (must be first Streamlit call)
# =========================================================
 
st.set_page_config(
    page_title="Mood AI",
    page_icon="🤖",
    layout="centered"
)
 
 
# =========================================================
# API KEY CHECK — fail loudly and clearly, not with a crash
# =========================================================
 
if not os.getenv("MISTRAL_API_KEY"):
    st.error(
        "MISTRAL_API_KEY not found. Add it to your .env file "
        "as MISTRAL_API_KEY=your_key_here and restart the app."
    )
    st.stop()
 
 
# =========================================================
# MODEL (cached so it's not rebuilt on every rerun)
# =========================================================
 
@st.cache_resource
def get_model():
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.9
    )
 
model = get_model()
 
 
# =========================================================
# CUSTOM CSS
# =========================================================
 
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        margin-bottom: 0px;
    }
    .developer {
        text-align: center;
        color: gray;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
 
# =========================================================
# TITLE
# =========================================================
 
st.markdown('<h1 class="main-title">🤖 Mood AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="developer">Developed by Dharmesh Sharma</p>', unsafe_allow_html=True)
 
 
# =========================================================
# AI MODES
# =========================================================
 
modes = {
    "😡 Angry Mode":
        "You are an Angry AI agent. You respond aggressively and impatiently, "
        "but you never use slurs, threats, or genuinely abusive language — "
        "keep it exaggerated and comedic, not actually hostile.",
 
    "😂 Funny Mode":
        "You are a Funny AI agent. You respond in a funny, joyful manner with "
        "good vibes, energy, humour and jokes.",
 
    "😢 Sad Mode":
        "You are a Sad AI agent. You respond in a sad and melancholic manner."
}
 
mode_avatars = {
    "😡 Angry Mode": "😡",
    "😂 Funny Mode": "😂",
    "😢 Sad Mode": "😢",
}
 
MAX_HISTORY_MESSAGES = 20  # trim old turns so tokens/cost don't balloon
 
 
# =========================================================
# INITIALIZE SESSION STATE (before anything reads it)
# =========================================================
 
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "😂 Funny Mode"
 
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=modes[st.session_state.selected_mode])]
 
 
# =========================================================
# SIDEBAR — just utility actions, no mode picker here
# =========================================================
 
with st.sidebar:
    st.subheader("Settings")
 
    if st.button("🗑️ New Chat"):
        st.session_state.messages = [SystemMessage(content=modes[st.session_state.selected_mode])]
        st.rerun()
 
    if len(st.session_state.messages) > 1:
        chat_text = "\n\n".join(
            f"{'You' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in st.session_state.messages
            if not isinstance(m, SystemMessage)
        )
        st.download_button("⬇️ Download Chat", chat_text, file_name="mood_ai_chat.txt")
 
 
# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================
 
current_avatar = mode_avatars[st.session_state.selected_mode]
 
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar=current_avatar):
            st.write(message.content)
 
 
# =========================================================
# BOTTOM MODE PICKER — small dropdown next to input,
# styled like Claude's model-selector pill (not a big
# selectbox sitting at the top of the page anymore)
# =========================================================
 
st.markdown(
    """
    <style>
    div[data-testid="stSelectbox"] {
        max-width: 180px;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 12px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
picker_col, _spacer = st.columns([1, 4])
 
with picker_col:
    selected_mode = st.selectbox(
        "Mode",
        options=list(modes.keys()),
        index=list(modes.keys()).index(st.session_state.selected_mode),
        label_visibility="collapsed"
    )
 
if selected_mode != st.session_state.selected_mode:
    st.session_state.selected_mode = selected_mode
 
    # Replace only the system message; keep prior turns visible.
    # Use the sidebar "New Chat" button if you want a full reset instead.
    for i, msg in enumerate(st.session_state.messages):
        if isinstance(msg, SystemMessage):
            st.session_state.messages[i] = SystemMessage(content=modes[selected_mode])
            break
 
    st.rerun()
 
 
# =========================================================
# CHAT INPUT — st.chat_input auto-clears after submit,
# supports Enter key, and sits pinned at the bottom
# =========================================================
 
prompt = st.chat_input("Message your AI...")
 
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
 
    with st.chat_message("user"):
        st.write(prompt)
 
    with st.chat_message("assistant", avatar=current_avatar):
        try:
            # Trim history sent to the model (keep system msg + last N turns)
            system_msg = st.session_state.messages[0]
            recent = st.session_state.messages[1:][-MAX_HISTORY_MESSAGES:]
            trimmed_context = [system_msg] + recent
 
            # Stream tokens live instead of waiting for the full response
            full_response = st.write_stream(
                chunk.content for chunk in model.stream(trimmed_context)
            )
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"Something went wrong talking to the model: {e}")