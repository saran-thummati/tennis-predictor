import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client
from datetime import date
import extra_streamlit_components as stx
import stripe

# --- 0. PAGE CONFIGURATION ---
st.set_page_config(page_title="Tennis Predictor", layout="centered")

# --- 1. SETUP STRIPE (CRASH-PROOF) ---
if "STRIPE_SECRET_KEY" in st.secrets:
    stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]
    stripe_configured = True
else:
    stripe_configured = False

# UPDATE THIS TO YOUR ACTUAL STREAMLIT URL
DOMAIN = "https://netrix.streamlit.app/" 

# --- 2. CONNECT TO SUPABASE ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

# --- 3. GOOGLE AUTH CATCHER ---
if "code" in st.query_params:
    try:
        auth_code = st.query_params["code"]
        # Trade the code for a secure Supabase session
        session = supabase.auth.exchange_code_for_session({"auth_code": auth_code})
        st.session_state.user = session.user
        
        # Check if they already exist in your tracking table
        user_check = supabase.table("user_usage").select("*").eq("user_id", session.user.id).execute()
        
        # If they are brand new, build their database tracker
        if len(user_check.data) == 0:
            supabase.table("user_usage").insert({
                "user_id": session.user.id,
                "email": session.user.email,
                "predictions_used": 0,
                "last_reset_date": str(date.today()),
                "subscription_tier": "Free"
            }).execute()
            
        # Clear the URL and refresh into the app
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Google Login Failed: {e}")

# --- 4. STRIPE PAYMENT CATCHER ---
if stripe_configured and "success" in st.query_params and "session_id" in st.query_params:
    session_id = st.query_params["session_id"]
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            payer_uid = session.client_reference_id
            supabase.table("user_usage").update({
                "subscription_tier": "Premium"
            }).eq("user_id", payer_uid).execute()
            st.balloons()
            st.success("Payment Successful! You are now a Premium Member.")
            st.query_params.clear()
    except Exception as e:
        st.error("Could not verify payment.")

# --- 5. SESSION & COOKIE MEMORY ---
cookie_manager = stx.CookieManager()

if "user" not in st.session_state:
    st.session_state.user = None
if "match_result" not in st.session_state:
    st.session_state.match_result = None

current_cookie = cookie_manager.get(cookie="anon_preds")
cookie_val = int(current_cookie) if current_cookie else 0

if "anon_preds" not in st.session_state:
    st.session_state.anon_preds = cookie_val
if cookie_val > st.session_state.anon_preds:
    st.session_state.anon_preds = cookie_val

# --- 6. LOAD MODELS ---
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

# --- 7. THE PAYWALL POPUP (DIALOG) ---
@st.dialog("Prediction Limit Reached")
def show_paywall():
    st.warning("You have run out of free predictions.")
    st.write("Join the Elite Predictor for $5/month to access unlimited daily math-backed insights.")
    
    if st.session_state.user is None:
        st.error("Please use the sidebar to Sign Up for a free account to unlock 50 daily predictions!")
    else:
        if stripe_configured:
            try:
                checkout_session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price_data': {
                            'currency': 'usd',
                            'unit_amount': 500, 
                            'recurring': {'interval': 'month'},
                            'product_data': {
                                'name': 'Elite Predictor Premium',
                                'description': 'Unlimited daily AI tennis predictions',
                            },
                        },
                        'quantity': 1,
                    }],
                    mode='subscription',
                    client_reference_id=st.session_state.user.id,
                    success_url=DOMAIN + '/?success=true&session_id={CHECKOUT_SESSION_ID}',
                    cancel_url=DOMAIN + '/?canceled=true',
                )
                st.link_button("💳 Pay with Stripe to Upgrade", checkout_session.url, type="primary", use_container_width=True)
            except Exception as e:
                st.error(f"Error connecting to Stripe: {e}")
        else:
            st.info("Payments are currently disabled. Check back later!")

# --- 8. SIDEBAR (Auth & Progress) ---
with st.sidebar:
    if st.session_state.user is None:
        st.header("Guest Mode")
        st.progress(st.session_state.anon_preds / 5.0, text=f"{st.session_state.anon_preds} / 5 Free Predictions")
        st.divider()
        st.subheader("Unlock 50 Predictions")
        st.write("Create a free account to track more matches!")
        
        # 1. Google Sign-In Button
        if st.button("🌐 Continue with Google", use_container_width=True):
            res = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {"redirect_to": DOMAIN}
            })
            st.markdown(f'<meta http-equiv="refresh" content="0;url={res.url}">', unsafe_allow_html=True)
            
        st.markdown("<p style='text-align: center; color: gray;'>OR</p>", unsafe_allow_html=True)
        
        # 2. Manual Email Sign-In
        auth_mode = st.radio("Choose Action", ["Log In", "Sign Up"], label_visibility="collapsed")
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
                            st.error(f"Error: Ensure email is unique.")
                    else:
                        try:
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            st.session_state.user = response.user
                            st.rerun()
                        except Exception as e:
                            st.error("Login failed. Check credentials.")
                            
    else:
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

        st.write(f"**Account:** {st.session_state.user.email}")
        st.write(f"**Tier:** {tier}")
        if tier == "Free":
            st.progress(preds_used / 50.0, text=f"{preds_used} / 50 Daily Predictions Used")
        else:
            st.success("Unlimited Predictions Unlocked")
            
        if st.button("Log Out"):
            st.session_state.user = None
            st.rerun()

# --- 9. MAIN PREDICTOR UI ---
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
        
        # 1. GUEST CHECK
        if st.session_state.user is None:
            if st.session_state.anon_preds >= 5:
                show_paywall() 
            else:
                can_predict = True
                st.session_state.anon_preds += 1
                cookie_manager.set("anon_preds", str(st.session_state.anon_preds), max_age=86400)
        
        # 2. LOGGED IN CHECK
        else:
            if tier == "Free" and preds_used >= 50:
                show_paywall() 
            else:
                can_predict = True
                supabase.table("user_usage").update({"predictions_used": preds_used + 1}).eq("user_id", user_id).execute()

        # 3. DO THE MATH
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

# --- 10. DRAW THE RESULT ---
if st.session_state.match_result:
    res = st.session_state.match_result
    st.subheader("Prediction")
    if res["p1_win_prob"] > 0.5:
        st.success(f"**Predicted Winner:** {res['p1']}")
        st.metric(label=f"{res['p1']} Win Probability", value=f"{res['p1_win_prob'] * 100:.1f}%")
    else:
        st.success(f"**Predicted Winner:** {res['p2']}")
        st.metric(label=f"{res['p2']} Win Probability", value=f"{(1 - res['p1_win_prob']) * 100:.1f}%")
