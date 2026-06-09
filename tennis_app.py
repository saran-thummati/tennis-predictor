import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import calendar
import requests
import plotly.express as px
import streamlit.components.v1 as components

# ---- Data Loading & Mock Models (Truncated for brevity, assuming existing ML logic is intact) ----
@st.cache_data(ttl=86400)
def load_data():
    return pd.DataFrame() # Replace with your actual load_data

def get_prediction(*args, **kwargs):
    return 0.65, {} # Replace with your actual prediction logic

# ---- App Initialization ----
st.set_page_config(page_title="Tennis Match Predictor", layout="wide", initial_sidebar_state="collapsed")

# 1. Custom CSS for the Hamburger Menu & Clickable Buttons
st.markdown("""
    <style>
    /* Change the sidebar arrow to a hamburger menu ☰ */
    [data-testid="collapsedControl"] svg {
        display: none !important;
    }
    [data-testid="collapsedControl"]::before {
        content: "☰";
        font-size: 26px;
        color: #FFFFFF;
        display: block;
        margin-left: 5px;
    }
    
    /* Hide the radio circles and style sidebar options */
    div[role="radiogroup"] > label > div:first-of-type { display: none !important; }
    div[role="radiogroup"] > label {
        margin-bottom: 15px; padding: 10px 15px; border-radius: 6px;
        cursor: pointer; font-size: 1.1rem; font-weight: 600; color: #E0E0E0;
    }
    div[role="radiogroup"] > label:hover { background-color: rgba(255, 255, 255, 0.08); }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: rgba(76, 175, 80, 0.15); color: #4CAF50; border-left: 4px solid #4CAF50;
    }
    
    /* Style the top scorecard buttons */
    .stButton > button {
        width: 100%; height: 100%; min-height: 100px;
        background-color: #1e2130; border: 1px solid #333; border-left: 5px solid #4CAF50;
        text-align: left; justify-content: flex-start; padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Session State Management ----
if "menu" not in st.session_state:
    st.session_state.menu = "Home"
if "selected_match" not in st.session_state:
    st.session_state.selected_match = None

# Sidebar Menu Setup (Removed the word "Navigation")
with st.sidebar:
    st.markdown("<br/>", unsafe_allow_html=True)
    # Map the radio button to the session state
    selected_menu = st.radio("", ["Home", "Matches & Calendar", "Predictor", "History Tracker"], 
                             index=["Home", "Matches & Calendar", "Predictor", "History Tracker"].index(st.session_state.menu),
                             label_visibility="collapsed")
    
    # If user clicks sidebar, reset selected match and update state
    if selected_menu != st.session_state.menu:
        st.session_state.menu = selected_menu
        st.session_state.selected_match = None
        st.rerun()

# ---- UI: Top Live Score Strip ----
st.markdown("### 🕒 Live & Upcoming Matches")
st.caption("Click any scorecard to view advanced analytics.")
score_cols = st.columns(4)

# We use buttons so they are clickable and update the session state
with score_cols[0]:
    if st.button("🎾 LIVE - Set 3\n\nJ. Sinner (6) (4) (3)\nA. Zverev (4) (6) (2)", key="top_m1"):
        st.session_state.selected_match = "Sinner vs Zverev"
        st.session_state.menu = "Matches & Calendar"
        st.rerun()
with score_cols[1]:
    if st.button("🎾 LIVE - Set 1\n\nC. Alcaraz (5)\nJ. Draper (4)", key="top_m2"):
        st.session_state.selected_match = "Alcaraz vs Draper"
        st.session_state.menu = "Matches & Calendar"
        st.rerun()
with score_cols[2]:
    if st.button("🕒 Today - 17:30\n\nD. Medvedev\nH. Rune", key="top_m3"):
        st.session_state.selected_match = "Medvedev vs Rune"
        st.session_state.menu = "Matches & Calendar"
        st.rerun()
with score_cols[3]:
    if st.button("🕒 Tomorrow - 13:00\n\nT. Fritz\nA. de Minaur", key="top_m4"):
        st.session_state.selected_match = "Fritz vs de Minaur"
        st.session_state.menu = "Matches & Calendar"
        st.rerun()
st.divider()

# ==================== PAGE: HOME (NEWS & X POSTS) ====================
if st.session_state.menu == "Home":
    st.title("Top Tennis Headlines")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Fixed Image Loading & Clickable Links
        st.image("https://images.unsplash.com/photo-1595435934249-5df7ed86e1c0?q=80&w=1000&auto=format&fit=crop", use_container_width=True, caption="Grass Court Season Heats Up")
        st.markdown("### [Alcaraz claims surface speed transitions favor aggressive playstyles](https://www.atptour.com)")
        st.write("An early assessment detailing layout configuration training routines ahead of major schedule turn points as the ATP tour converges on Queen's Club.")
        st.caption("ESPN Tennis • 1 Hour Ago")

    with col2:
        st.subheader("Trending Articles")
        st.markdown("**[Sinner looks to stabilize return metrics after clinical run](https://www.bleacherreport.com)**")
        st.caption("Bleacher Report • 2 Hours Ago")
        st.divider()
        st.markdown("**[Djokovic confirms recovery periods ahead of Wimbledon](https://www.tennis.com)**")
        st.caption("Tennis.com • 4 Hours Ago")
        
    st.divider()
    st.subheader("What's Buzzing on X (Twitter)")
    # Embedding X posts dynamically using HTML components
    x_col1, x_col2, x_col3 = st.columns(3)
    with x_col1:
        components.html("""
        <blockquote class="twitter-tweet"><p lang="en" dir="ltr">The grass court season is officially here! 🌱🎾 Let the slides and slices begin. <a href="https://twitter.com/atptour/status/1798361730032644268"></a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        """, height=400)
    with x_col2:
        components.html("""
        <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Carlos Alcaraz arriving at Queen&#39;s Club looking ready to defend his title. 👑 <a href="https://twitter.com/TennisTV/status/1798361730032644268"></a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        """, height=400)

# ==================== PAGE: MATCHES & CALENDAR ====================
elif st.session_state.menu == "Matches & Calendar":
    
    # If a match was clicked from the top strip or the calendar, show the detail view
    if st.session_state.selected_match:
        match_title = st.session_state.selected_match
        st.button("← Back to Calendar", on_click=lambda: st.session_state.update({"selected_match": None}))
        st.title(f"Match Breakdown: {match_title}")
        
        # Display Stats Table
        st.markdown("### Comparative Statistics")
        html = """
        <table style='width:100%; text-align:center; border-collapse: collapse;'>
            <tr style='border-bottom: 2px solid #333;'><th style='text-align:left; padding:10px;'>Statistic</th><th>Player 1</th><th>Player 2</th></tr>
            <tr style='border-bottom: 1px solid #222;'><td style='text-align:left; padding:10px;'>Recent Win Form %</td><td style='background-color: rgba(76, 175, 80, 0.25); font-weight: bold;'>82.5%</td><td>76.0%</td></tr>
            <tr style='border-bottom: 1px solid #222;'><td style='text-align:left; padding:10px;'>1st Serve Win %</td><td>75.4%</td><td style='background-color: rgba(76, 175, 80, 0.25); font-weight: bold;'>78.2%</td></tr>
            <tr style='border-bottom: 1px solid #222;'><td style='text-align:left; padding:10px;'>Unforced Errors</td><td style='background-color: rgba(76, 175, 80, 0.25); font-weight: bold;'>14</td><td>21</td></tr>
        </table><br/>
        """
        st.markdown(html, unsafe_allow_html=True)
        
        # Win Probability Timeline & Pie Chart
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Win Probability Timeline")
            time_steps = [f"Set 1 Gm {i}" for i in range(1, 11)]
            df_time = pd.DataFrame({"Match Phase": time_steps, "Player 1 Prob": [50, 55, 52, 60, 65, 62, 70, 75, 72, 80]}).set_index("Match Phase")
            st.line_chart(df_time)
        with c2:
            st.markdown("### Current Match Projection")
            fig = px.pie(values=[80, 20], names=['Player 1', 'Player 2'], color_discrete_sequence=['#4CAF50', '#2a2d3d'], hole=0.5)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # Otherwise, show the visual Grid Calendar
    else:
        st.title("📅 Tournament Calendar")
        st.caption("June 2026")
        
        # Build the calendar header
        days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        cols = st.columns(7)
        for i, day in enumerate(days_of_week):
            cols[i].markdown(f"**{day}**")
        
        # Build the visual grid (Mocking June 2026 dates)
        # June 1, 2026 is a Monday
        day_counter = 1
        for week in range(5):
            grid_cols = st.columns(7)
            for i in range(7):
                if day_counter <= 30:
                    with grid_cols[i]:
                        # Draw a box for the day
                        st.markdown(f"<div style='border: 1px solid #444; border-radius: 5px; padding: 5px; min-height: 100px;'><b>{day_counter}</b>", unsafe_allow_html=True)
                        
                        # Inject matches on specific days
                        if day_counter == 8: # Today
                            if st.button("Sinner vs Zverev", key="cal_m1"):
                                st.session_state.selected_match = "Sinner vs Zverev"
                                st.rerun()
                        elif day_counter == 9:
                            if st.button("Fritz vs de Minaur", key="cal_m2"):
                                st.session_state.selected_match = "Fritz vs de Minaur"
                                st.rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    day_counter += 1

# ==================== PAGE: PREDICTOR & HISTORY ====================
elif st.session_state.menu == "Predictor":
    st.title("Custom Match Predictor")
    st.write("Predictor Logic goes here...")

elif st.session_state.menu == "History Tracker":
    st.title("Session History Tracker")
    st.write("History logs go here...")
