import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
from io import BytesIO
import os

# ===== CONFIG =====
DEFAULT_BACKEND_URL = os.getenv("MUSIC_BACKEND_URL", "http://127.0.0.1:5000")

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
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if 'last_health_check' not in st.session_state:
    st.session_state.last_health_check = 0.0
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'active_model' not in st.session_state:
    st.session_state.active_model = None

# ===== FUNCTIONS =====

def check_backend(force=False):
    now = time.time()
    if not force and (now - st.session_state.last_health_check) < 5:
        return st.session_state.backend_connected
    try:
        r = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
        if r.status_code == 200:
            st.session_state.backend_connected = True
            st.session_state.last_health_check = now
            payload = r.json()
            st.session_state.active_model = payload.get("active_model")
            return True
    except requests.RequestException:
        st.session_state.last_health_check = now
    st.session_state.backend_connected = False
    return False


def generate_music(seed, notes, tempo, temperature, top_k, top_p, repeat_penalty, apply_smoothing, variable_rhythm):
    try:
        r = requests.post(
            f"{st.session_state.backend_url}/generate/lstm",
            json={
                'seed': seed,
                'num_notes': notes,
                'tempo': tempo,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repeat_penalty': repeat_penalty,
                'apply_smoothing': apply_smoothing,
                'variable_rhythm': variable_rhythm,
            },
            timeout=60
        )
        if r.status_code >= 400:
            return {'error': r.json().get('error', f"Request failed: {r.status_code}")}
        return r.json()
    except requests.RequestException as e:
        return {'error': str(e)}


def get_models():
    try:
        r = requests.get(f"{st.session_state.backend_url}/models", timeout=5)
        r.raise_for_status()
        payload = r.json()
        st.session_state.available_models = payload.get("models", [])
        st.session_state.active_model = payload.get("active_model")
        return True
    except requests.RequestException:
        st.session_state.available_models = []
        return False


def select_model(model_file):
    try:
        r = requests.post(
            f"{st.session_state.backend_url}/models/select",
            json={"model_file": model_file},
            timeout=20
        )
        if r.status_code >= 400:
            return {'error': r.json().get('error', f"Request failed: {r.status_code}")}
        payload = r.json()
        st.session_state.active_model = payload.get("active_model")
        return payload
    except requests.RequestException as e:
        return {'error': str(e)}


@st.cache_data(ttl=4, show_spinner=False)
def fetch_history(backend_url):
    r = requests.get(f"{backend_url}/history", timeout=5)
    r.raise_for_status()
    return r.json().get("history", [])


def get_history():
    try:
        return fetch_history(st.session_state.backend_url)
    except requests.RequestException:
        return []


@st.cache_data(show_spinner=False)
def fetch_midi_bytes(backend_url, filename):
    r = requests.get(f"{backend_url}/download/{filename}", timeout=10)
    r.raise_for_status()
    return r.content


def download_midi(filename):
    try:
        return BytesIO(fetch_midi_bytes(st.session_state.backend_url, filename))
    except requests.RequestException:
        return None


# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Controls")
    st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)

    if st.button("🔌 Connect Backend"):
        check_backend(force=True)
        get_models()

    if st.button("🔄 Refresh History"):
        fetch_history.clear()

    if st.session_state.backend_connected:
        st.success("Backend Connected")
    else:
        st.error("Backend Not Connected")

    st.markdown("### 🧠 Model")
    if st.session_state.backend_connected and not st.session_state.available_models:
        get_models()

    if st.session_state.available_models:
        model_index = 0
        if st.session_state.active_model in st.session_state.available_models:
            model_index = st.session_state.available_models.index(st.session_state.active_model)
        selected_model = st.selectbox(
            "Select .h5 model",
            st.session_state.available_models,
            index=model_index
        )
        if st.button("Load Selected Model"):
            response = select_model(selected_model)
            if 'error' in response:
                st.error(response['error'])
            else:
                st.success(f"Active model: {response.get('active_model')}")
    else:
        st.info("No .h5 models found yet.")

    seed = st.number_input("Seed", 0, 9999, 42)
    num_notes = st.slider("Notes", 10, 200, 50)
    tempo = st.slider("Tempo", 60, 200, 120)
    st.markdown("### 🎛️ Sampling")
    temperature = st.slider("Temperature", 0.1, 1.8, 0.8, 0.05)
    top_k = st.slider("Top-K", 0, 100, 0, 1)
    top_p = st.slider("Top-P", 0.1, 1.0, 1.0, 0.05)
    repeat_penalty = st.slider("Repeat Penalty", 0.0, 2.0, 0.0, 0.1)
    apply_smoothing = st.checkbox("Apply pitch smoothing", value=True)
    variable_rhythm = st.checkbox("Enable variable rhythm", value=True)

# ===== MAIN =====
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #111827 0%, #0b1220 45%, #070b13 100%);
    }
    .hero-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 18px 20px;
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(6px);
    }
    .stat-chip {
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 999px;
        padding: 6px 12px;
        margin-right: 10px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎵 AI Music Studio")
check_backend()

if not st.session_state.backend_connected:
    st.info("Start the Flask backend (`python server.py`) and then click Connect Backend.")
    st.stop()

st.session_state.generation_history = get_history()
active_model_label = st.session_state.active_model or "Not loaded"
st.markdown(
    f"""
    <div class="hero-card">
        <h3 style="margin:0 0 10px 0;">Generate expressive MIDI with your selected model</h3>
        <span class="stat-chip">Backend: Connected</span>
        <span class="stat-chip">Active Model: {active_model_label}</span>
        <span class="stat-chip">History: {len(st.session_state.generation_history)}</span>
    </div>
    """,
    unsafe_allow_html=True
)

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
            res = generate_music(
                seed,
                num_notes,
                tempo,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
                apply_smoothing,
                variable_rhythm
            )

        if 'error' in res:
            st.error(res['error'])
        else:
            st.success("Generated!")
            fetch_history.clear()
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

        mood = latest.get('metrics', {}).get(
            'mood',
            {'emoji': '🎼', 'primary': 'balanced', 'confidence': 0}
        )
        st.markdown(
            f"### {mood['emoji']} Mood: {mood['primary'].title()}  \n"
            f"Confidence: {mood['confidence']}%"
        )

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
                "Gen": i + 1,
                "Pitch": m.get('avg_pitch', 0),
                "Range": m.get('pitch_range', 0),
                "Mood": m.get('mood', {}).get('primary', '')
            })

        df = pd.DataFrame(data)

        fig = px.scatter(
            df,
            x="Gen",
            y="Pitch",
            size="Range",
            color="Mood",
            title="Generation Trends: Average Pitch vs Range"
        )
        fig.update_layout(xaxis_title="Generation", yaxis_title="Average Pitch")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: HISTORY
# =========================

with tab3:
    history = list(reversed(st.session_state.generation_history))
    page_size = 10
    total_pages = max(1, (len(history) + page_size - 1) // page_size)
    page = st.number_input("History Page", min_value=1, max_value=total_pages, value=1, step=1)
    page_index = int(page) - 1
    start = page_index * page_size
    end = start + page_size

    for idx, g in enumerate(history[start:end], start=1):
        title = f"Run {start + idx} - Seed {g['seed']} - Notes {g['num_notes']} - Tempo {g['tempo']}"
        with st.expander(title):
            st.caption(f"Model used: {g.get('model_file', 'unknown')}")
            if st.button("Load Audio", key=f"load_audio_{start + idx}"):
                midi = download_midi(g['midi_file'])
                if midi:
                    st.audio(midi, format='audio/midi')
                else:
                    st.warning("Could not download MIDI for this run.")