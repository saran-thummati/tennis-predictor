import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client
from datetime import date
import extra_streamlit_components as stx
import plotly.express as px

# --- 0. PAGE CONFIGURATION ---
# We must set it to "wide" so the 3/4 map and 1/4 sidebar look proportional
st.set_page_config(page_title="Elite Tennis Predictor", layout="wide")

# --- 1. CONNECT TO SUPABASE ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

# --- 2. SESSION & COOKIE MEMORY ---
cookie_manager = stx.CookieManager()

if "user" not in st.session_state:
    st.session_state.user = None
if "match_result" not in st.session_state:
    st.session_state.match_result = None
# Tracks if they clicked "Use as Guest"
if "is_guest" not in st.session_state:
    st.session_state.is_guest = False
# Tells the landing page whether to show "Log In" or "Sign Up" initially
if "auth_action" not in st.session_state:
    st.session_state.auth_action = "Log In"

current_cookie = cookie_manager.get(cookie="anon_preds")
anon_preds = int(current_cookie) if current_cookie else 0

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

# --- 4. THE PAYWALL POPUP (DIALOG) ---
@st.dialog("Prediction Limit Reached")
def show_paywall():
    st.warning("You have run out of free predictions. Sign in to unlock more.")
    st.write("Join the Elite Predictor to access unlimited daily math-backed insights.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In", use_container_width=True):
            st.session_state.is_guest = False
            st.session_state.auth_action = "Log In"
            st.rerun()
    with col2:
        if st.button("Sign Up", type="primary", use_container_width=True):
            st.session_state.is_guest = False
            st.session_state.auth_action = "Sign Up"
            st.rerun()


# ==========================================
# VIEW 1: THE LANDING PAGE (Map + Auth)
# ==========================================
if st.session_state.user is None and not st.session_state.is_guest:
    
    # Create the 3/4 and 1/4 layout
    map_col, auth_col = st.columns([3, 1])
    
    with map_col:
        st.title("Global Tennis Analytics")
        # Define the Grand Slam coordinates
        slams = pd.DataFrame({
            "City": ["New York", "London", "Paris", "Melbourne"],
            "Lat": [40.7128, 51.5074, 48.8566, -37.8136],
            "Lon": [-74.0060, -0.1278, 2.3522, 144.9631],
            "Tournament": ["US Open", "Wimbledon", "Roland Garros", "Australian Open"]
        })
        
        # Build the custom black & white map
        fig = px.scatter_geo(slams, lat="Lat", lon="Lon", hover_name="Tournament")
        fig.update_geos(
            showcountries=True, countrycolor="white",
            showland=True, landcolor="black",
            showocean=True, oceancolor="#111111", # Slightly lighter black for ocean contrast
            bgcolor="black",
            projection_type="natural earth"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", # Transparent background to match Streamlit
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        # Style the points
        fig.update_traces(marker=dict(size=12, color="white", line=dict(width=2, color="gray")))
        
        st.plotly_chart(fig, use_container_width=True)

    with auth_col:
        st.header("Access the AI")
        
        # The Auth Form
        auth_mode = st.radio("Account", ["Log In", "Sign Up"], index=0 if st.session_state.auth_action == "Log In" else 1)
        
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
                            supabase.auth.sign_up({"email": email, "password": password})
                            login_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            st.session_state.user = login_response.user
                            supabase.table("user_usage").insert({
                                "user_id": login_response.user.id,
                                "email": email,
                                "predictions_used": 0,
                                "last_reset_date": str(date.today()),
                                "subscription_tier": "Free"
                            }).execute()
                            st.rerun()
                        except Exception as e:
                            st.error("Error creating account. Ensure email is unique.")
                    else:
                        try:
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            st.session_state.user = response.user
                            st.rerun()
                        except Exception as e:
                            st.error("Login failed. Check credentials.")
                            
        st.divider()
        st.markdown("<p style='text-align: center; color: gray;'>Or try it before you buy it</p>", unsafe_allow_html=True)
        if st.button("Use as Guest", use_container_width=True):
            st.session_state.is_guest = True
            st.rerun()

# ==========================================
# VIEW 2: THE MAIN PREDICTOR APP
# ==========================================
else:
    # Check Database Limits for Logged-In Users
    if st.session_state.user is not None:
        user_id = st.session_state.user.id
        today_str = str(date.today())
        
        usage_data = supabase.table("user_usage").select("*").eq("user_id", user_id).execute()
        
        if len(usage_data.data) > 0:
            user_record = usage_data.data[0]
            preds_used = user_record["predictions_used"]
            last_reset = user_record["last_reset_date"]
            tier = user_record["subscription_tier"]
            
            if last_reset != today_str:
                supabase.table("user_usage").update({"predictions_used": 0, "last_reset_date": today_str}).eq("user_id", user_id).execute()
                preds_used = 0
        else:
            preds_used = 0
            tier = "Free"

    # Minimal Sidebar
    with st.sidebar:
        if st.session_state.user is not None:
            st.write(f"**Account:** {st.session_state.user.email}")
            st.write(f"**Tier:** {tier}")
            if st.button("Log Out"):
                st.session_state.user = None
                st.rerun()
        else:
            st.write("**Account:** Guest Mode")
            if st.button("Exit Guest Mode"):
                st.session_state.is_guest = False
                st.rerun()

    # The Predictor UI
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
            
            # 1. Guest Verification (Pop the Dialog if limit reached)
            if st.session_state.user is None:
                if anon_preds >= 5:
                    show_paywall()  # TRIGGERS THE POPUP
                else:
                    can_predict = True
                    new_count = anon_preds + 1
                    cookie_manager.set("anon_preds", str(new_count), max_age=86400)
                    anon_preds = new_count 
                    
            # 2. Logged In User Verification
            else:
                if tier == "Free" and preds_used >= 50:
                    st.error("You have used all 50 free predictions for today. Come back tomorrow.")
                else:
                    can_predict = True
                    supabase.table("user_usage").update({"predictions_used": preds_used + 1}).eq("user_id", user_id).execute()

            # Execute Prediction if approved
            if can_predict:
                with st.spinner("Calculating Probabilities..."):
                    
                    # Core Math
                    r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
                    r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
                    p1_wins_h2h = h2h_tracker.get(f"{p1}_vs_{p2}", 0)
                    p2_wins_h2h = h2h_tracker.get(f"{p2}_vs_{p1}", 0)
                    h2h_adv = p1_wins_h2h - p2_wins_h2h
                    
                    # Momentum Multiplier implementation (From our earlier fix)
                    p1_recent = surface_form.get(p1, {}).get(surface, [0.5])
                    p2_recent = surface_form.get(p2, {}).get(surface, [0.5])
                    form_1_surf = (sum(p1_recent) / len(p1_recent)) * 100 if p1_recent else 50.0
                    form_2_surf = (sum(p2_recent) / len(p2_recent)) * 100 if p2_recent else 50.0
                    
                    momentum_multiplier = 3.0 
                    form_adv = ((form_1_surf / 100) - (form_2_surf / 100)) * momentum_multiplier
                    
                    b1 = player_bio.get(p1, {"age": 25.0, "hand": "R"})
                    b2 = player_bio.get(p2, {"age": 25.0, "hand": "R"})
                    p1_is_lefty = 1 if b1["hand"] == 'L' else 0
                    p2_is_lefty = 1 if b2["hand"] == 'L' else 0

                    features = np.array([[r1_base - r2_base, r1_surf - r2_surf, h2h_adv, form_adv, b1["age"] - b2["age"], p1_is_lefty - p2_is_lefty]])
                    probabilities = ai_model.predict_proba(features)[0]
                    p1_win_prob = probabilities[1] 
                    
                    st.session_state.match_result = {
                        "p1": p1, "p2": p2, "p1_win_prob": p1_win_prob
                    }
                    st.rerun()

    # Draw the Prediction Result
    if st.session_state.match_result:
        res = st.session_state.match_result
        st.subheader("Prediction")
        if res["p1_win_prob"] > 0.5:
            st.success(f"**Predicted Winner:** {res['p1']}")
            st.metric(label=f"{res['p1']} Win Probability", value=f"{res['p1_win_prob'] * 100:.1f}%")
        else:
            st.success(f"**Predicted Winner:** {res['p2']}")
            st.metric(label=f"{res['p2']} Win Probability", value=f"{(1 - res['p1_win_prob']) * 100:.1f}%")
