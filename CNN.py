import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Reshape, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

from LoadingData import get_df


# Load the data
df = get_df()

# Filter for specific themes
contains_mate_in_2 = df['Themes'].str.contains('mateIn2', na=False)
contains_backrank_mate = df['Themes'].str.contains('backRankMate', na=False)
m2_and_backrank = df[contains_mate_in_2 & contains_backrank_mate].copy()


### Preparing the data

# Encoding pieces for the chess board
def piece_to_number(piece):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
              'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
    return pieces.get(piece, 0)

# Parse the FEN string into a numeric matrix
def parse_fen(fen):
    board = np.zeros((8, 8))
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)  # Move over empty squares
            else:
                board[i, col] = piece_to_number(char)
                col += 1
    return board

# Apply the parse_fen to each FEN string and create board matrices
m2_and_backrank['board_matrix'] = m2_and_backrank['FEN'].apply(parse_fen)

# Prepare X as a numpy array of reshaped board matrices
X = np.array(m2_and_backrank['board_matrix'].tolist()).reshape((-1, 8, 8, 1))

# Encode and pad moves
all_moves = [move for sublist in m2_and_backrank['Moves'].apply(lambda x: x.split()) for move in sublist]
label_encoder = LabelEncoder()
label_encoder.fit(all_moves)
m2_and_backrank['Encoded Moves'] = m2_and_backrank['Moves'].apply(lambda x: label_encoder.transform(x.split()))
max_sequence_length = max(m2_and_backrank['Encoded Moves'].apply(len))
y = pad_sequences(m2_and_backrank['Encoded Moves'], maxlen=max_sequence_length, padding='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Reshape((max_sequence_length, -1)),  # Reshape for LSTM
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(len(label_encoder.classes_), activation='softmax'))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

# Save the trained model
model.save('chess_puzzle_solver_model.h5')


### Testing the model

# Make predictions
predictions = model.predict(X_test)

# Convert probabilities to class indices
predicted_indices = np.argmax(predictions, axis=-1)
# Use the label encoder to transform indices back to move notations
predicted_moves = [label_encoder.inverse_transform(pred) for pred in predicted_indices]

def calculate_accuracy(true_sequences, predicted_sequences):
    correct_count = 0
    total_count = len(true_sequences)

    for true_seq, pred_seq in zip(true_sequences, predicted_sequences):
        # Assuming padding values are zeros in integer-encoded sequences
        true_seq = [move for move in true_seq if move != 0]  # Adjust depending on your actual padding value
        # Compare lengths and actual moves
        if len(true_seq) == len(pred_seq) and all(t == p for t, p in zip(true_seq, pred_seq)):
            correct_count += 1

    return correct_count / total_count

# Decode y_test from categorical to actual moves for comparison
true_moves = [label_encoder.inverse_transform(np.trim_zeros(y)) for y in y_test]

# Calculate the accuracy
accuracy = calculate_accuracy(true_moves, predicted_moves)
print(f"Accuracy of solving puzzles: {accuracy * 100:.2f}%")




# Accuracy per move

def progressive_accuracy(true_sequences, predicted_sequences):
    # Initialize a dictionary to store the results
    max_length = max(len(seq) for seq in true_sequences)
    correct_predictions = {i: 0 for i in range(1, max_length + 1)}
    
    # Iterate over each pair of true and predicted sequences
    for true_seq, pred_seq in zip(true_sequences, predicted_sequences):
        # Convert to list if they are numpy arrays to ensure comparisons are element-wise and not array-wise
        if isinstance(true_seq, np.ndarray):
            true_seq = true_seq.tolist()
        if isinstance(pred_seq, np.ndarray):
            pred_seq = pred_seq.tolist()
        
        # Truncate pred_seq to the length of true_seq to avoid index out of range errors
        pred_seq = pred_seq[:len(true_seq)]

        # Check up to the length of the true sequence
        for i in range(1, len(true_seq) + 1):
            if true_seq[:i] == pred_seq[:i]:
                correct_predictions[i] += 1

    return correct_predictions

# Assuming true_moves and predicted_moves are lists of moves already prepared
# Example, true_moves might be: [['e2e4', 'e7e5'], ['d2d4', 'd7d5'], ...]
# predicted_moves might be: [['e2e4', 'e7e6'], ['d2d4', 'd7d5'], ...]

# Calculate progressive accuracy
progressive_results = progressive_accuracy(true_moves, predicted_moves)

# Print the results
for length in sorted(progressive_results.keys()):
    print(f"Correct predictions for the first {length} moves: {progressive_results[length]} out of {len(true_moves)} puzzles = {progressive_results[length]/len(true_moves)*100:.2f}%")







### Approach using opponents moves

# Load the data
df = get_df()

# Filter for specific themes
contains_mate_in_2 = df['Themes'].str.contains('mateIn2', na=False)
contains_backrank_mate = df['Themes'].str.contains('backRankMate', na=False)
m2_and_backrank = df[contains_mate_in_2 & contains_backrank_mate].copy()


# Function to encode chess pieces to numeric values
def piece_to_number(piece):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
              'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
    return pieces.get(piece, 0)

# Function to parse FEN strings into numeric matrices
def parse_fen(fen):
    board = np.zeros((8, 8), dtype=int)
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board[i, col] = piece_to_number(char)
                col += 1
    return board

# Apply parse_fen to each FEN string
m2_and_backrank['board_matrix'] = m2_and_backrank['FEN'].apply(parse_fen)
X = np.array(m2_and_backrank['board_matrix'].tolist()).reshape((-1, 8, 8, 1))

# Encode all unique moves using LabelEncoder
all_moves = sorted(set(move for moves in df['Moves'] for move in moves.split()))
label_encoder = LabelEncoder()
label_encoder.fit(all_moves)

# Encode moves
def encode_moves(moves):
    move_indices = label_encoder.transform(moves.split())
    return to_categorical(move_indices, num_classes=len(label_encoder.classes_))

# Apply encoding to moves
m2_and_backrank['Encoded Moves'] = m2_and_backrank['Moves'].apply(encode_moves)
max_sequence_length = max(len(encoded) for encoded in m2_and_backrank['Encoded Moves'])
y = np.stack(m2_and_backrank['Encoded Moves'].values)

# Prepare the initial board state and target first move
initial_board_states = np.array([parse_fen(fen) for fen in df['FEN']])
first_moves_encoded = np.array([encode_moves(moves.split()[1]) for moves in df['Moves']])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(initial_board_states, first_moves_encoded, test_size=0.2, random_state=42)

# Build and compile the model
model = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Reshape((max_sequence_length, -1)),  # Reshape for LSTM
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(len(label_encoder.classes_), activation='softmax'))
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)




