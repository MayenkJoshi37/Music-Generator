# =========================
# CLEAN LOGS (TOP OF FILE)
# =========================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# IMPORTS
# =========================
from create import load_notes, prepare_sequences, generate_notes
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import numpy as np
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
import logging

from keras.models import load_model
from music21 import stream, note, chord, instrument, tempo

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
    'n_vocab': None,
    'active_model': None,
}

generation_history = []
MAX_HISTORY = 200

# =========================
# LOAD MODEL + DATA ONCE
# =========================

def _model_files():
    return sorted(
        [f for f in os.listdir(".") if f.lower().endswith(".h5") and os.path.isfile(f)]
    )


def _load_model_file(model_name):
    available = _model_files()
    safe_name = os.path.basename(model_name)
    if safe_name not in available:
        raise FileNotFoundError(f"Model file not found: {safe_name}")
    models['lstm_model'] = load_model(safe_name)
    models['active_model'] = safe_name
    return safe_name


print("🚀 Loading model and dataset...")

try:
    notes = load_notes("dataset/*.mid")
    network_input, pitchnames = prepare_sequences(notes)

    models['network_input'] = network_input
    models['pitchnames'] = pitchnames
    models['n_vocab'] = len(pitchnames)

    available_models = _model_files()
    preferred = "final_model.h5"
    default_model = preferred if preferred in available_models else (available_models[0] if available_models else None)
    if default_model:
        _load_model_file(default_model)
        print(f"✅ Model loaded: {default_model}")
    else:
        print("⚠️ No .h5 model files found in project directory")

    print("✅ Dataset prepared successfully")

except Exception as e:
    print(f"❌ Error loading model/data: {e}")

# =========================
# UTILS
# =========================

def notes_to_midi_bytes(pitches, durations, tempo_val=120):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=tempo_val))
    s.append(instrument.Piano())

    for pitch_group, dur in zip(pitches, durations):
        try:
            if isinstance(pitch_group, list) and len(pitch_group) > 1:
                notes = [note.Note(int(p)) for p in pitch_group]
                n = chord.Chord(notes)
            else:
                val = pitch_group[0] if isinstance(pitch_group, list) else pitch_group
                n = note.Note(int(val))
            n.quarterLength = float(dur)
            s.append(n)
        except (TypeError, ValueError):
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


def _parse_generation_params(payload):
    try:
        seed = int(payload.get('seed', 42))
        num_notes = int(payload.get('num_notes', 50))
        tempo_val = int(payload.get('tempo', 120))
        temperature = float(payload.get('temperature', 0.8))
        top_k = int(payload.get('top_k', 0))
        top_p = float(payload.get('top_p', 1.0))
        repeat_penalty = float(payload.get('repeat_penalty', 0.0))
        apply_smoothing = bool(payload.get('apply_smoothing', True))
        variable_rhythm = bool(payload.get('variable_rhythm', True))
    except (TypeError, ValueError):
        raise ValueError("Invalid parameter types supplied")

    if not (0 <= seed <= 10_000_000):
        raise ValueError("seed must be between 0 and 10000000")
    if not (10 <= num_notes <= 250):
        raise ValueError("num_notes must be between 10 and 250")
    if not (40 <= tempo_val <= 240):
        raise ValueError("tempo must be between 40 and 240")
    if not (0.1 <= temperature <= 1.8):
        raise ValueError("temperature must be between 0.1 and 1.8")
    if not (0 <= top_k <= 100):
        raise ValueError("top_k must be between 0 and 100")
    if not (0.1 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.1 and 1.0")
    if not (0.0 <= repeat_penalty <= 2.0):
        raise ValueError("repeat_penalty must be between 0.0 and 2.0")

    return {
        'seed': seed,
        'num_notes': num_notes,
        'tempo': tempo_val,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'repeat_penalty': repeat_penalty,
        'apply_smoothing': apply_smoothing,
        'variable_rhythm': variable_rhythm,
    }


def _token_to_pitch_group(token):
    token_text = str(token)
    if '.' in token_text:
        pitch_values = []
        for item in token_text.split('.'):
            try:
                pitch_values.append(int(item))
            except (TypeError, ValueError):
                continue
        return pitch_values if pitch_values else [60]
    try:
        n = note.Note(token_text)
        return [n.pitch.midi]
    except Exception:
        return [60]


def _smooth_pitch_groups(pitch_groups):
    if not pitch_groups:
        return pitch_groups
    smoothed = [pitch_groups[0]]
    prev_anchor = int(np.mean(pitch_groups[0]))
    for group in pitch_groups[1:]:
        current_anchor = int(np.mean(group))
        gap = current_anchor - prev_anchor
        if abs(gap) > 14:
            shift = -12 if gap > 0 else 12
            group = [max(36, min(96, p + shift)) for p in group]
            current_anchor = int(np.mean(group))
        smoothed.append(group)
        prev_anchor = current_anchor
    return smoothed


def _generate_durations(length, variable_rhythm):
    if not variable_rhythm:
        return [0.5] * length
    palette = [0.25, 0.5, 0.75, 1.0]
    probs = [0.15, 0.5, 0.15, 0.2]
    return [float(np.random.choice(palette, p=probs)) for _ in range(length)]


def _pitch_groups_for_metrics(pitch_groups):
    if not pitch_groups:
        return []
    return [int(np.mean(group)) for group in pitch_groups if group]

# =========================
# GENERATION
# =========================

def generate_with_lstm(params):
    import random

    seed = params['seed']
    num_notes = params['num_notes']
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
        length=num_notes,
        temperature=params['temperature'],
        top_k=params['top_k'],
        top_p=params['top_p'],
        repeat_penalty=params['repeat_penalty']
    )

    pitch_groups = [_token_to_pitch_group(el) for el in prediction_output]
    if params['apply_smoothing']:
        pitch_groups = _smooth_pitch_groups(pitch_groups)
    durations = _generate_durations(len(pitch_groups), params['variable_rhythm'])
    metric_pitches = _pitch_groups_for_metrics(pitch_groups)

    return pitch_groups, durations, metric_pitches

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
        'history_count': len(generation_history),
        'active_model': models['active_model']
    })


@app.route('/models')
def list_models():
    available = _model_files()
    return jsonify({
        'models': available,
        'active_model': models['active_model']
    })


@app.route('/models/select', methods=['POST'])
def select_model():
    payload = request.json or {}
    model_name = payload.get('model_file')
    if not model_name:
        return jsonify({'error': 'model_file is required'}), 400
    try:
        loaded = _load_model_file(model_name)
        return jsonify({'success': True, 'active_model': loaded})
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        app.logger.exception("Model selection failed")
        return jsonify({'error': f'Could not load model: {e}'}), 500


@app.route('/generate/lstm', methods=['POST'])
def generate_lstm():
    if models['lstm_model'] is None:
        return jsonify({'error': 'Model not loaded yet'}), 503

    data = request.json or {}
    try:
        params = _parse_generation_params(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        pitch_groups, durations, metric_pitches = generate_with_lstm(params)

        metrics = analyze_music(metric_pitches, durations)
        midi_bytes = notes_to_midi_bytes(pitch_groups, durations, params['tempo'])

        filename = f"lstm_{datetime.now().strftime('%H%M%S')}.mid"
        filepath = os.path.join(GENERATED_FOLDER, filename)

        with open(filepath, 'wb') as f:
            f.write(midi_bytes)

        generation_history.append({
            'model': 'LSTM',
            'model_file': models['active_model'],
            'seed': params['seed'],
            'num_notes': params['num_notes'],
            'tempo': params['tempo'],
            'metrics': metrics,
            'midi_file': filename,
            'sampling': {
                'temperature': params['temperature'],
                'top_k': params['top_k'],
                'top_p': params['top_p'],
                'repeat_penalty': params['repeat_penalty'],
                'apply_smoothing': params['apply_smoothing'],
                'variable_rhythm': params['variable_rhythm'],
            }
        })
        if len(generation_history) > MAX_HISTORY:
            del generation_history[:-MAX_HISTORY]

        return jsonify({
            'success': True,
            'metrics': metrics,
            'midi_file': filename,
            'sampling': generation_history[-1]['sampling']
        })

    except Exception as e:
        app.logger.exception("Generation failed")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        abort(400)
    path = os.path.abspath(os.path.join(GENERATED_FOLDER, safe_name))
    base = os.path.abspath(GENERATED_FOLDER)
    if not path.startswith(base):
        abort(403)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


@app.route('/history')
def history():
    return jsonify({'history': generation_history})

# =========================
# RUN
# =========================

if __name__ == '__main__':
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    print("🎵 Backend running at http://127.0.0.1:5000")
    app.run(debug=False)  # IMPORTANT