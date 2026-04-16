import glob
import numpy as np
import random
from music21 import converter, instrument, note, chord, stream
from keras.models import load_model
import tensorflow as tf
from collections import defaultdict

def get_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Using GPU")
            return "/GPU:0"
        except:
            pass
    print("⚠️ Using CPU")
    return "/CPU:0"

DEVICE = get_device()
# =========================
# 1. LOAD NOTES (same as training)
# =========================

def load_notes(dataset_path):
    notes = []
    for file in glob.glob(dataset_path):
        try:
            midi = converter.parse(file)
        except:
            continue

        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


# =========================
# 2. PREPARE INPUT (same mapping)
# =========================

def prepare_sequences(notes, sequence_length=50):
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])

    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    network_input = network_input / float(len(pitchnames))

    return network_input, pitchnames


# =========================
# 3. SAMPLING (important)
# =========================

def _apply_top_k(probs, top_k):
    if top_k is None or top_k <= 0 or top_k >= len(probs):
        return probs
    top_indices = np.argpartition(probs, -top_k)[-top_k:]
    filtered = np.zeros_like(probs)
    filtered[top_indices] = probs[top_indices]
    total = np.sum(filtered)
    return filtered / total if total > 0 else probs


def _apply_top_p(probs, top_p):
    if top_p is None or top_p <= 0 or top_p >= 1:
        return probs
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, top_p, side="left") + 1
    keep_idx = sorted_idx[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[keep_idx] = probs[keep_idx]
    total = np.sum(filtered)
    return filtered / total if total > 0 else probs


def sample(preds, temperature=0.8, top_k=0, top_p=1.0):
    temp = max(0.1, float(temperature))
    scaled = np.log(preds + 1e-8) / temp
    exp_preds = np.exp(scaled - np.max(scaled))
    probs = exp_preds / np.sum(exp_preds)
    probs = _apply_top_k(probs, int(top_k) if top_k else 0)
    probs = _apply_top_p(probs, float(top_p) if top_p else 1.0)
    return np.random.choice(len(probs), p=probs)


# =========================
# 4. GENERATE NOTES
# =========================

def generate_notes(
    model,
    network_input,
    pitchnames,
    n_vocab,
    length=200,
    temperature=0.8,
    top_k=0,
    top_p=1.0,
    repeat_penalty=0.0
):
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    start = random.randint(0, len(network_input)-1)
    pattern = network_input[start]

    prediction_output = []
    generated_counts = defaultdict(int)

    for _ in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))

        prediction = model.predict(prediction_input, verbose=0)
        probs = prediction[0].copy()
        if repeat_penalty and repeat_penalty > 0:
            for idx, count in generated_counts.items():
                probs[idx] = probs[idx] / (1.0 + (repeat_penalty * count))
            probs = probs / np.sum(probs)

        index = sample(
            probs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        result = int_to_note[index]
        prediction_output.append(result)
        generated_counts[index] += 1

        pattern = np.append(pattern, index / float(n_vocab))
        pattern = pattern[1:]

    return prediction_output


# =========================
# 5. CREATE MIDI
# =========================

def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:

        if pattern is None or pattern == "":
            continue

        if '.' in pattern:
            try:
                notes_in_chord = pattern.split('.')
                notes = [note.Note(int(n)) for n in notes_in_chord]
                chord_obj = chord.Chord(notes)
                chord_obj.offset = offset
                output_notes.append(chord_obj)
            except:
                continue
        else:
            try:
                new_note = note.Note(pattern)
                new_note.offset = offset
                output_notes.append(new_note)
            except:
                continue

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='generated_output.mid')

# =========================
# 6. MAIN
# =========================

if __name__ == "__main__":
    print("Loading model...")
    model = load_model("final_model.h5")

    print("Loading dataset...")
    notes = load_notes("dataset/*.mid")

    print("Preparing input...")
    network_input, pitchnames = prepare_sequences(notes)

    print("Generating music...")
    prediction_output = generate_notes(
        model,
        network_input,
        pitchnames,
        len(pitchnames)
    )

    print("Saving MIDI...")
    create_midi(prediction_output)

    print("Done. Output: generated_output.mid")