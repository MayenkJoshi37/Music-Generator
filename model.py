import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

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
# 1. LOAD DATA
# =========================

def load_notes(dataset_path):
    notes = []
    for file in glob.glob(dataset_path):
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"Skipping file {file} due to error")
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
# 2. PREPARE DATA
# =========================

def prepare_sequences(notes, sequence_length=50):
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))

    network_output = to_categorical(network_output)

    return network_input, network_output, pitchnames


# =========================
# 3. MODEL
# =========================

def create_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# =========================
# 4. GENERATE MUSIC
# =========================

def sample(preds, temperature=1.0):
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


# =========================
# 5. CONVERT TO MIDI
# =========================

def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            chord_obj = chord.Chord(notes)
            chord_obj.offset = offset
            output_notes.append(chord_obj)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')


# =========================
# 6. MAIN
# =========================

if __name__ == "__main__":
    print("Loading data...")
    notes = load_notes("dataset/*.mid")

    print("Preparing sequences...")
    network_input, network_output, pitchnames = prepare_sequences(notes)

    print("Creating model...")
    model = create_model(
        (network_input.shape[1], network_input.shape[2]),
        network_output.shape[1]
    )

    print("Training...")
    
    checkpoint = ModelCheckpoint(
        "model.h5",
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    model.fit(
        network_input,
        network_output,
        epochs=30,
        batch_size=128,
        callbacks=[checkpoint]
    )
    
    # Save final model also
    model.save("final_model.h5")

    print("Done.")