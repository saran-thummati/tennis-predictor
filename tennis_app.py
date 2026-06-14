import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client
from elo_model import EloModel 

# --- 1. CONNECT TO SUPABASE ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

# --- 2. SESSION STATE ---
if "user" not in st.session_state:
    st.session_state.user = None

# --- 3. THE LOGIN / SIGNUP PAGE ---
if st.session_state.user is None:
    st.set_page_config(page_title="Tennis Predictor", layout="centered")
    st.title("Tennis Predictor")
    st.markdown("Create a free account to access the AI.")
    
          tab1, tab2 = st.tabs(["Log In", "Sign Up"])
    
    with tab1:
        st.subheader("Welcome Back")
        # Use a form to safely capture the login data
        with st.form("login_form"):
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Log In", use_container_width=True)
            
            if submit_login:
                if not login_email or not login_password:
                    st.error("Please fill in both fields.")
                else:
                    try:
                        response = supabase.auth.sign_in_with_password({"email": login_email, "password": login_password})
                        st.session_state.user = response.user
                        st.rerun()
                    except Exception as e:
                        st.error("Login failed. Check your email and password.")

    with tab2:
        st.subheader("Create a Free Account")
        # Use a form to safely capture the signup data
        with st.form("signup_form"):
            signup_email = st.text_input("Email")
            signup_password = st.text_input("Password", type="password")
            submit_signup = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_signup:
                if not signup_email or not signup_password:
                    st.error("Please fill in both fields.")
                else:
                    try:
                        response = supabase.auth.sign_up({"email": signup_email, "password": signup_password})
                        st.success("Account created successfully! You can now switch to the Log In tab.")
                    except Exception as e:
                        st.error(f"Error creating account: {e}")
                        

# --- 4. THE MAIN APP (Protected) ---
else:
    @st.cache_resource
    def load_model_artifacts():
        return joblib.load("tennis_model_artifacts.pkl")

    try:
        artifacts = load_model_artifacts()
        all_players = artifacts["all_players"]
        model_elo = artifacts["model_elo"]
        ai_model = artifacts["ai_model"]
        h2h_tracker = artifacts["h2h_tracker"]
        surface_form = artifacts.get("surface_form", {}) 
        player_bio = artifacts["player_bio"]
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.user.email}**")
        if st.button("Log Out"):
            st.session_state.user = None
            st.rerun()

    st.title("Tennis predictor")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Select First Player", all_players, key="p1")
    with col2:
        default_p2_idx = 1 if len(all_players) > 1 else 0
        p2 = st.selectbox("Select Second Player", all_players, index=default_p2_idx, key="p2")

    st.divider()
    surface = st.selectbox("Select Court Surface", ["Hard", "Clay", "Grass"])

    if st.button("Predict Match Outcome", use_container_width=True):
        if p1 == p2:
            st.warning("Please select two different players.")
        else:
            with st.spinner("Calculating Probabilities..."):
                r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
                r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
                p1_wins_h2h = h2h_tracker.get(f"{p1}_vs_{p2}", 0)
                p2_wins_h2h = h2h_tracker.get(f"{p2}_vs_{p1}", 0)
                h2h_adv = p1_wins_h2h - p2_wins_h2h
                p1_recent = surface_form.get(p1, {}).get(surface, [0.5])
                p2_recent = surface_form.get(p2, {}).get(surface, [0.5])
                form_1_surf = (sum(p1_recent) / len(p1_recent)) * 100 if p1_recent else 50.0
                form_2_surf = (sum(p2_recent) / len(p2_recent)) * 100 if p2_recent else 50.0
                form_adv = (form_1_surf / 100) - (form_2_surf / 100)
                b1 = player_bio.get(p1, {"age": 25.0, "hand": "R"})
                b2 = player_bio.get(p2, {"age": 25.0, "hand": "R"})
                p1_is_lefty = 1 if b1["hand"] == 'L' else 0
                p2_is_lefty = 1 if b2["hand"] == 'L' else 0

                features = np.array([[r1_base - r2_base, r1_surf - r2_surf, h2h_adv, form_adv, b1["age"] - b2["age"], p1_is_lefty - p2_is_lefty]])
                probabilities = ai_model.predict_proba(features)[0]
                p1_win_prob = probabilities[1] 
                
                st.subheader("Prediction")
                if p1_win_prob > 0.5:
                    st.success(f"**Predicted Winner:** {p1}")
                    st.metric(label=f"{p1} Win Probability", value=f"{p1_win_prob * 100:.1f}%")
                else:
                    st.success(f"**Predicted Winner:** {p2}")
                    st.metric(label=f"{p2} Win Probability", value=f"{(1 - p1_win_prob) * 100:.1f}%")
