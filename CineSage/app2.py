import os
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="CineSage",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# API KEY
# =========================================================

if not os.getenv("MISTRAL_API_KEY"):
    st.error("MISTRAL_API_KEY is missing from your .env file.")
    st.stop()


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def get_model():
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.2
    )


model = get_model()


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an information extraction AI.

Analyze the following movie paragraph and extract useful information from it.

Extract:

- Movie Name
- Genre
- Main Characters
- Cast (only if mentioned)
- Director (only if mentioned)
- Release Year (only if mentioned)
- Setting
- Plot
- Main Themes
- Central Conflict
- Key Relationships
- Overall Tone
- Quick Summary

Rules:

- Extract only information supported by the paragraph.
- Do not guess or invent missing information.
- If something is not mentioned, write "Not mentioned".
- Keep the extracted information concise.
- Give the Quick Summary in 2-3 sentences.

Return the information in a clean, readable format.
"""
    ),
    (
        "human",
        """
Extract useful information from the following movie description:

{paragraph}
"""
    )
])


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 2rem;
}

/* Hero */

.hero {
    text-align: center;
    padding: 10px 0 35px 0;
}

.hero-icon {
    font-size: 42px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-top: 5px;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 16px;
    color: #9ca3af;
}

.developer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: -22px;
    margin-bottom: 35px;
}


/* Input */

.input-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

textarea {
    border-radius: 14px !important;
}


/* Button */

.stButton > button {
    border-radius: 12px;
    height: 46px;
    font-weight: 600;
}


/* Result */

.result-header {
    font-size: 20px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 15px;
}

.result-container {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 16px;
    padding: 24px;
}


/* Footer */

.footer {
    text-align: center;
    color: #777;
    font-size: 12px;
    padding-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🎬</div>
        <div class="hero-title">CineSage</div>
        <div class="hero-subtitle">
            Turn movie descriptions into meaningful insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="developer">Developed by Dharmesh Sharma</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT
# =========================================================

st.markdown(
    '<div class="input-title">📝 Movie Description</div>',
    unsafe_allow_html=True
)

movie_text = st.text_area(
    "movie_description",
    placeholder=(
        "Paste a movie description, plot, review, "
        "or any useful information about a movie..."
    ),
    height=220,
    label_visibility="collapsed"
)


# =========================================================
# EXTRACT BUTTON
# =========================================================

_, button_col, _ = st.columns([1, 2, 1])

with button_col:

    extract = st.button(
        "✨ Extract Information",
        use_container_width=True
    )


# =========================================================
# EXTRACTION
# =========================================================

if extract:

    if not movie_text.strip():

        st.warning("Please enter a movie description first.")

    else:

        with st.spinner("Analyzing your movie description..."):

            try:

                final_prompt = prompt.invoke({
                    "paragraph": movie_text
                })

                response = model.invoke(final_prompt)

                st.session_state.result = response.content

            except Exception as e:

                st.error(f"Something went wrong: {e}")


# =========================================================
# RESULT
# =========================================================

if "result" in st.session_state:

    st.markdown(
        '<div class="result-header">✨ Extracted Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-container">',
        unsafe_allow_html=True
    )

    st.markdown(st.session_state.result)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if st.button("🗑️ Clear Result"):
        del st.session_state.result
        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        CineSage · AI-powered movie information extraction
        <br><br>
        Developed by Dharmesh Sharma
    </div>
    """,
    unsafe_allow_html=True
)