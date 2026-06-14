import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client
from datetime import date

# --- 1. CONNECT TO SUPABASE ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

# --- 2. SESSION STATE (MEMORY) ---
if "user" not in st.session_state:
    st.session_state.user = None
# This tracks our anonymous users!
if "anon_preds" not in st.session_state:
    st.session_state.anon_preds = 0

# --- 3. LOAD MODELS ---
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
    st.error("Could not load model artifacts. Check your files.")
    st.stop()

# --- 4. SIDEBAR LOGIC (Auth & Limits) ---
with st.sidebar:
    # IF NOT LOGGED IN
    if st.session_state.user is None:
        st.header("Guest Mode")
        st.progress(st.session_state.anon_preds / 5.0, text=f"{st.session_state.anon_preds} / 5 Free Predictions")
        st.divider()
        st.subheader("Unlock 50 Predictions")
        st.write("Create a free account to track more matches!")
        
        # Sidebar Login/Signup Form
        auth_mode = st.radio("Choose Action", ["Log In", "Sign Up"])
        with st.form("auth_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button(auth_mode, use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("Fill all fields.")
                else:
                    if auth_mode == "Sign Up":
                        try:
                            # 1. Sign Up
                            supabase.auth.sign_up({"email": email, "password": password})
                            # 2. Auto Log In
                            login_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            st.session_state.user = login_response.user
                            # 3. Build Database Tracker
                            supabase.table("user_usage").insert({
                                "user_id": login_response.user.id,
                                "email": email,
                                "predictions_used": 0,
                                "last_reset_date": str(date.today()),
                                "subscription_tier": "Free"
                            }).execute()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        try:
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            st.session_state.user = response.user
                            st.rerun()
                        except Exception as e:
                            st.error("Login failed. Check credentials.")
                            
    # IF LOGGED IN
    else:
        user_id = st.session_state.user.id
        today_str = str(date.today())
        
        usage_data = supabase.table("user_usage").select("*").eq("user_id", user_id).execute()
        
        if len(usage_data.data) > 0:
            user_record = usage_data.data[0]
            preds_used = user_record["predictions_used"]
            last_reset = user_record["last_reset_date"]
            tier = user_record["subscription_tier"]
            
            # Reset daily limit
            if last_reset != today_str:
                supabase.table("user_usage").update({"predictions_used": 0, "last_reset_date": today_str}).eq("user_id", user_id).execute()
                preds_used = 0
        else:
            preds_used = 0
            tier = "Free"

        st.write(f"**Account:** {st.session_state.user.email}")
        st.write(f"**Tier:** {tier}")
        if tier == "Free":
            st.progress(preds_used / 50.0, text=f"{preds_used} / 50 Daily Predictions Used")
        else:
            st.success("Unlimited Predictions Unlocked")
            
        if st.button("Log Out"):
            st.session_state.user = None
            st.rerun()

# --- 5. MAIN PREDICTOR APP ---
st.title("Elite Tennis Predictor")
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
        can_predict = False
        
        # Check Limits BEFORE doing math
        if st.session_state.user is None:
            if st.session_state.anon_preds >= 5:
                st.error("You have used your 5 free guest predictions! Please use the sidebar to create a free account and unlock 50 more.")
            else:
                can_predict = True
                st.session_state.anon_preds += 1
        else:
            if tier == "Free" and preds_used >= 50:
                st.error("You have used all 50 free predictions for today. Come back tomorrow or upgrade to Premium.")
            else:
                can_predict = True
                supabase.table("user_usage").update({"predictions_used": preds_used + 1}).eq("user_id", user_id).execute()

        # Execute Prediction if approved
        if can_predict:
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
