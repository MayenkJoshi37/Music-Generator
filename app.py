import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
from io import BytesIO

# ===== CONFIG =====
BACKEND_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="wide"
)

# ===== SESSION =====
if 'backend_connected' not in st.session_state:
    st.session_state.backend_connected = False
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []

# ===== FUNCTIONS =====

def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            st.session_state.backend_connected = True
            return True
    except:
        pass
    st.session_state.backend_connected = False
    return False


def generate_music(seed, notes, tempo):
    try:
        r = requests.post(
            f"{BACKEND_URL}/generate/lstm",
            json={'seed': seed, 'num_notes': notes, 'tempo': tempo},
            timeout=60
        )
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def get_history():
    try:
        r = requests.get(f"{BACKEND_URL}/history")
        return r.json().get("history", [])
    except:
        return []


def download_midi(filename):
    try:
        r = requests.get(f"{BACKEND_URL}/download/{filename}")
        return BytesIO(r.content)
    except:
        return None


def show_mood(mood):
    st.markdown(f"""
    ### {mood['emoji']} Mood: {mood['primary'].title()}
    Confidence: {mood['confidence']}%
    """)


# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔌 Connect Backend"):
        check_backend()

    if st.session_state.backend_connected:
        st.success("Backend Connected")
    else:
        st.error("Backend Not Connected")

    seed = st.number_input("Seed", 0, 9999, 42)
    num_notes = st.slider("Notes", 10, 200, 50)
    tempo = st.slider("Tempo", 60, 200, 120)

# ===== MAIN =====

st.title("🎵 AI Music Generator (LSTM)")

if not st.session_state.backend_connected:
    st.stop()

st.session_state.generation_history = get_history()

tab1, tab2, tab3 = st.tabs([
    "🎹 Generate",
    "📊 Analytics",
    "📚 History"
])

# =========================
# TAB 1: GENERATE
# =========================

with tab1:
    if st.button("Generate Music"):
        with st.spinner("Generating..."):
            res = generate_music(seed, num_notes, tempo)

        if 'error' in res:
            st.error(res['error'])
        else:
            st.success("Generated!")

            time.sleep(1)
            st.rerun()

    if st.session_state.generation_history:
        latest = st.session_state.generation_history[-1]

        midi = download_midi(latest['midi_file'])

        if midi:
            st.audio(midi, format='audio/midi')
            st.download_button(
                "Download MIDI",
                midi.getvalue(),
                file_name=latest['midi_file']
            )

        if 'mood' in latest['metrics']:
            mood = latest['metrics']['mood']

        st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(135deg, #6C63FF, #4ECDC4);
            color: white;
            text-align: center;
        ">
            <h1>{mood['emoji']} {mood['primary'].title()}</h1>
            <p style="font-size: 18px;">Confidence: {mood['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)

        m = latest['metrics']
        st.markdown("### 📊 Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎵 Pitch Range", f"{m.get('pitch_range', 0)}")

        with col2:
            st.metric("🔁 Repetition", f"{m.get('repetition_factor', 0):.2f}")

        with col3:
            st.metric("🎼 Unique Notes", f"{m.get('unique_pitches', 0)}")

# =========================
# TAB 2: ANALYTICS
# =========================

with tab2:
    if not st.session_state.generation_history:
        st.info("Generate music first")
    else:
        data = []

        for i, g in enumerate(st.session_state.generation_history):
            m = g['metrics']
            data.append({
                "Gen": i,
                "Pitch": m.get('avg_pitch', 0),
                "Range": m.get('pitch_range', 0),
                "Mood": m.get('mood', {}).get('primary', '')
            })

        df = pd.DataFrame(data)

        fig = px.scatter(df, x="Gen", y="Pitch", size="Range", color="Mood")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: HISTORY
# =========================

with tab3:
    for g in reversed(st.session_state.generation_history):
        st.markdown(f"""
        **Seed:** {g['seed']}  
        **Notes:** {g['num_notes']}  
        **Tempo:** {g['tempo']}
        """)

        midi = download_midi(g['midi_file'])

        if midi:
            st.audio(midi, format='audio/midi')

        st.markdown("---")