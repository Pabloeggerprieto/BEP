import numpy as np
import chess
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tqdm import tqdm
from Data.LoadingData import DataLoader

# Initialize DataLoader with the file path
loader = DataLoader('Data/lichess_db_puzzle.csv.zst')

# Load data
df = loader.backrank_and_m2()
df

# Function to update board after applying a move
def update_board(fen, move):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
    return board.fen()

# Function to parse FEN string into numeric matrix
def parse_fen(fen):
    board = np.zeros((8, 8), dtype=int)
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

# Function to encode chess piece to number
def piece_to_number(piece):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, 
              'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, 
              '.': 0}
    return pieces.get(piece, 0)

# Function to encode move to number
def move_to_number(move, label_encoder):
    return label_encoder.transform([move])[0] if move else None

# Label encoding for moves
all_moves = set()
for moves in df['Moves']:
    all_moves.update(moves.split())
label_encoder = LabelEncoder()
label_encoder.fit(list(all_moves))

# Prepare data for predicting second and fourth moves
def prepare_data(df, label_encoder):
    # Update board to the state after the opponent's first move
    df['FEN_after_first_move'] = df.apply(lambda row: update_board(row['FEN'], row['Moves'].split()[0]), axis=1)
    df['FEN_after_third_move'] = df.apply(lambda row: update_board(row['FEN_after_first_move'], row['Moves'].split()[2]), axis=1)
    
    # Encode the second and fourth moves
    df['encoded_second_move'] = [move_to_number(moves.split()[1], label_encoder) for moves in df['Moves']]
    df['encoded_fourth_move'] = [move_to_number(moves.split()[3], label_encoder) if len(moves.split()) > 3 else None for moves in df['Moves']]
    return df

df = prepare_data(df, label_encoder)

# Model to predict the second move
X_first = np.array(df['FEN_after_first_move'].apply(parse_fen).tolist()).reshape((-1, 8, 8, 1))
y_second = np.array(df['encoded_second_move'].tolist())
X_train_first, X_test_first, y_train_second, y_test_second = train_test_split(X_first, y_second, test_size=0.2, random_state=42)

model_first_move = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model_first_move.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_first_move.fit(X_train_first, y_train_second, epochs=10, batch_size=16, validation_split=0.2)   ## Change Batch size

# Model to predict the fourth move
X_second = np.array(df['FEN_after_third_move'].apply(parse_fen).tolist()).reshape((-1, 8, 8, 1))
y_fourth = np.array(df['encoded_fourth_move'].tolist())
X_train_second, X_test_second, y_train_fourth, y_test_fourth = train_test_split(X_second, y_fourth, test_size=0.2, random_state=42)

model_second_move = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model_second_move.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_second_move.fit(X_train_second, y_train_fourth, epochs=10, batch_size=16, validation_split=0.2)   ## Change Batch size



# Initialize the new columns
df['first_move_correct'] = False
df['second_move_correct'] = False

# Extract the actual second and fourth moves
df['actual_second_move'] = df['Moves'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)
df['actual_fourth_move'] = df['Moves'].apply(lambda x: x.split()[3] if len(x.split()) > 3 else None)

# Initialize the columns for the predicted moves
df['predicted_second_move'] = None  # will be updated during predictions
df['predicted_fourth_move'] = None  # will be updated during predictions


# Predicting moves with the first model and updating with the second model
correct_first_moves = 0
correct_second_moves = 0
total_moves = 0
correct_full_sequence = 0

# Prediction loop, assuming the data has been split and models are trained
for i in tqdm(range(len(X_test_first)), desc="Predicting moves"): ## Change length to len(X_test_first)
    board_state = X_test_first[i]
    actual_first_move = y_test_second[i]

    predicted_first_move_idx = np.argmax(model_first_move.predict(board_state.reshape(1, 8, 8, 1)))
    predicted_first_move = label_encoder.inverse_transform([predicted_first_move_idx])[0]

    if predicted_first_move == label_encoder.inverse_transform([actual_first_move])[0]:
        df.at[df.index[i], 'first_move_correct'] = True
        correct_first_moves += 1

        # Predict the second move (fourth chess move)
        board_state_second = X_test_second[i]
        actual_second_move = y_test_fourth[i]

        predicted_second_move_idx = np.argmax(model_second_move.predict(board_state_second.reshape(1, 8, 8, 1)))
        predicted_second_move = label_encoder.inverse_transform([predicted_second_move_idx])[0]

        # Store the predicted second move
        df.at[df.index[i], 'predicted_second_move'] = predicted_first_move
        df.at[df.index[i], 'predicted_fourth_move'] = predicted_second_move

        if predicted_second_move == label_encoder.inverse_transform([actual_second_move])[0]:
            df.at[df.index[i], 'second_move_correct'] = True
            correct_second_moves += 1
            correct_full_sequence += 1

    total_moves += 1



# Calculate accuracies
accuracy_first_move = correct_first_moves / total_moves if total_moves > 0 else 0
accuracy_second_move = correct_second_moves / correct_first_moves if total_moves > 0 else 0
accuracy_total = correct_full_sequence / total_moves if total_moves > 0 else 0

# Print results
print(f"Accuracy for predicting our second move (our first move in the sequence): {accuracy_first_move * 100 :.2f}%")
print(f"Accuracy for predicting our fourth move (our second move in the sequence): {accuracy_second_move* 100 :.2f}%")
print(f"Accuracy for predicting entire move: {accuracy_total* 100 :.2f}%")


### Analysing first move predictions success and failures

# df.columns
# good_first_move = df[['GameUrl', 'first_move_correct', 'predicted_second_move', 'actual_second_move']]
# good_first_move.head(20)


# df.iloc[49]


# bad_first_move = df[df['first_move_correct'] == False]
# bad_first_move


# good_second_move = df[df['second_move_correct'] == True]
# bad_second_move = df[df['second_move_correct'] == False]
