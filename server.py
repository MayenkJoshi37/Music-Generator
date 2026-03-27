# =========================
# CLEAN LOGS (TOP OF FILE)
# =========================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF logs

import warnings
warnings.filterwarnings("ignore")  # suppress all warnings

# =========================
# IMPORTS
# =========================
from create import load_notes, prepare_sequences, generate_notes
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import tempfile
from datetime import datetime

from keras.models import load_model
from music21 import stream, note, instrument, tempo

# =========================
# APP INIT
# =========================
app = Flask(__name__)
CORS(app)

GENERATED_FOLDER = 'generated_music'
os.makedirs(GENERATED_FOLDER, exist_ok=True)

models = {
    'lstm_model': None,
    'network_input': None,
    'pitchnames': None,
    'n_vocab': None
}

generation_history = []

# =========================
# LOAD MODEL + DATA ONCE
# =========================

print("🚀 Loading model and dataset...")

try:
    models['lstm_model'] = load_model("final_model.h5")

    notes = load_notes("dataset/*.mid")
    network_input, pitchnames = prepare_sequences(notes)

    models['network_input'] = network_input
    models['pitchnames'] = pitchnames
    models['n_vocab'] = len(pitchnames)

    print("✅ Model & dataset loaded successfully")

except Exception as e:
    print(f"❌ Error loading model/data: {e}")

# =========================
# UTILS
# =========================

def notes_to_midi_bytes(pitches, durations, tempo_val=120):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=tempo_val))
    s.append(instrument.Piano())

    for pitch, dur in zip(pitches, durations):
        try:
            n = note.Note(int(pitch))
            n.quarterLength = float(dur)
            s.append(n)
        except:
            continue

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        path = tmp.name

    s.write('midi', fp=path)

    with open(path, 'rb') as f:
        midi_bytes = f.read()

    os.remove(path)
    return midi_bytes

# =========================
# ANALYSIS
# =========================

def analyze_music(pitches, durations):
    if len(pitches) < 2:
        return {}

    numeric_notes = np.array(pitches)
    diffs = np.abs(np.diff(numeric_notes))

    metrics = {
        'avg_note_length': float(np.mean(durations)),
        'pitch_range': int(np.max(numeric_notes) - np.min(numeric_notes)),
        'avg_pitch': float(np.mean(numeric_notes)),
        'num_notes': len(numeric_notes),
        'unique_pitches': len(np.unique(numeric_notes)),
        'repetition_factor': float(1 - (len(np.unique(numeric_notes)) / len(numeric_notes))),
        'avg_pitch_change': float(np.mean(diffs)) if len(diffs) > 0 else 0,
    }

    metrics['mood'] = classify_mood(metrics)
    return metrics


def classify_mood(metrics):
    avg_pitch = metrics.get('avg_pitch', 60)
    repetition = metrics.get('repetition_factor', 0)

    if avg_pitch > 65:
        return {'primary': 'happy', 'emoji': '😊', 'confidence': 80}
    elif avg_pitch < 60:
        return {'primary': 'sad', 'emoji': '😢', 'confidence': 80}
    elif repetition < 0.3:
        return {'primary': 'energetic', 'emoji': '⚡', 'confidence': 80}
    else:
        return {'primary': 'calm', 'emoji': '😌', 'confidence': 80}

# =========================
# GENERATION
# =========================

def generate_with_lstm(seed, num_notes):
    import random

    np.random.seed(seed)
    random.seed(seed)

    model = models['lstm_model']
    network_input = models['network_input']
    pitchnames = models['pitchnames']
    n_vocab = models['n_vocab']

    prediction_output = generate_notes(
        model,
        network_input,
        pitchnames,
        n_vocab,
        length=num_notes
    )

    pitches = []
    for el in prediction_output:
        try:
            if '.' in str(el):
                pitches.append(int(el.split('.')[0]))
            else:
                n = note.Note(el)
                pitches.append(n.pitch.midi)
        except:
            pitches.append(60)

    durations = [0.5] * len(pitches)

    return pitches, durations

# =========================
# API
# =========================

@app.route('/')
def home():
    return jsonify({
        'status': 'online'
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'lstm_loaded': models['lstm_model'] is not None,
        'history_count': len(generation_history)
    })


@app.route('/generate/lstm', methods=['POST'])
def generate_lstm():
    data = request.json
    seed = data.get('seed', 42)
    num_notes = data.get('num_notes', 50)
    tempo_val = data.get('tempo', 120)

    try:
        pitches, durations = generate_with_lstm(seed, num_notes)

        metrics = analyze_music(pitches, durations)
        midi_bytes = notes_to_midi_bytes(pitches, durations, tempo_val)

        filename = f"lstm_{datetime.now().strftime('%H%M%S')}.mid"
        filepath = os.path.join(GENERATED_FOLDER, filename)

        with open(filepath, 'wb') as f:
            f.write(midi_bytes)

        generation_history.append({
            'model': 'LSTM',
            'seed': seed,
            'num_notes': num_notes,
            'tempo': tempo_val,
            'metrics': metrics,
            'midi_file': filename
        })

        return jsonify({
            'success': True,
            'metrics': metrics,
            'midi_file': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(GENERATED_FOLDER, filename), as_attachment=True)


@app.route('/history')
def history():
    return jsonify({'history': generation_history})

# =========================
# RUN
# =========================

if __name__ == '__main__':
    print("🎵 Backend running at http://127.0.0.1:5000")
    app.run(debug=False)  # IMPORTANT