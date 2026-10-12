"""Theme management for Streamlit UI (Light & Dark modes).

Persists theme state in st.session_state and injects custom CSS tokens
without Markdown indentation artifacts.
"""

from pathlib import Path
import streamlit as st


def get_current_theme() -> str:
    """Gets the active theme name ('dark' or 'light') from session state.

    Returns:
        str: Active theme name.
    """
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    return st.session_state["theme"]


def toggle_theme():
    """Toggles theme between dark and light mode."""
    current = get_current_theme()
    st.session_state["theme"] = "light" if current == "dark" else "dark"


def apply_theme():
    """Reads styles.css and injects theme styles into Streamlit page cleanly."""
    current = get_current_theme()
    css_path = Path(__file__).parent / "styles.css"

    css_content = ""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

    if current == "dark":
        dynamic_rules = """
.stApp {
    background-color: #0B0F19 !important;
    color: #F9FAFB !important;
}
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1F2937 !important;
}
header[data-testid="stHeader"] {
    background-color: rgba(11, 15, 25, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
    background-color: #1F2937 !important;
    color: #F9FAFB !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
}
.stButton button {
    border-radius: 8px !important;
}
[data-testid="stExpander"] {
    background-color: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 10px !important;
}
"""
    else:
        dynamic_rules = """
.stApp {
    background-color: #F9FAFB !important;
    color: #111827 !important;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
header[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
}
.stButton button {
    border-radius: 8px !important;
}
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
}
"""

    combined_style = f"<style>\n{css_content}\n{dynamic_rules}\n</style>"

    # Use st.html in Streamlit 1.35+ to avoid Markdown parsing entirely
    if hasattr(st, "html"):
        st.html(combined_style)
    else:
        st.markdown(combined_style, unsafe_allow_html=True)
