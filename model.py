import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf
from keras import mixed_precision
import time

# =========================
# TRAINING CONFIG
# =========================
SEQUENCE_LENGTH = 50
MAX_EPOCHS = 60
BATCH_SIZE = 128
MAX_TRAIN_MINUTES = 28  # keep under ~30 mins wall-time

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

if DEVICE == "/GPU:0":
    # Faster matmul on RTX cards; keeps final softmax dense output in float32.
    mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled")


class TimeLimitCallback(Callback):
    def __init__(self, max_minutes=28):
        super().__init__()
        self.max_seconds = max_minutes * 60
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"\n⏱️ Time limit reached ({self.max_seconds/60:.0f} min). Stopping training.")
            self.model.stop_training = True
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

def prepare_sequences(notes, sequence_length=SEQUENCE_LENGTH):
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
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim, activation='softmax', dtype='float32'))

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03)
    model.compile(loss=loss, optimizer='adam')
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
    print(f"Loaded {len(notes)} notes")
    print(f"Vocabulary size: {len(pitchnames)}")
    print(f"Training patterns: {network_input.shape[0]}")

    print("Creating model...")
    model = create_model(
        (network_input.shape[1], network_input.shape[2]),
        network_output.shape[1]
    )

    print("Training...")
    
    checkpoint = ModelCheckpoint(
        "model.h5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
    time_limit = TimeLimitCallback(max_minutes=MAX_TRAIN_MINUTES)
    
    model.fit(
        network_input,
        network_output,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[checkpoint, early_stop, lr_scheduler, time_limit],
        verbose=1
    )
    
    # Save final model also
    model.save("final_model.h5")

    print("Done.")