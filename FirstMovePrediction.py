from LoadingData import m2_and_backrank
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import chess

df = m2_and_backrank()  # Your data

# Assuming df['Moves'] contains the full move sequence and df['FEN'] is the initial board state
def prepare_data_for_second_move(df):
    # Updates the board to the state after the first move (opponent's move)
    df['FEN_after_first_move'] = df.apply(lambda row: update_board(row['FEN'], row['Moves'].split()[0]), axis=1)
    # The target move is now the second move in the sequence
    df['target_move'] = df['Moves'].apply(lambda moves: moves.split()[1] if len(moves.split()) > 1 else None)
    return df

def update_board(fen, move):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
    return board.fen()

def piece_to_number(piece):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, 'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
    return pieces.get(piece, 0)

# Parse the FEN string into a numeric matrix
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

df = prepare_data_for_second_move(df)

# Encoding the board and moves
df['board_matrix'] = df['FEN_after_first_move'].apply(parse_fen)
X = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))

# Label encoding moves
all_moves = set()
for fen in df['FEN_after_first_move']:
    board = chess.Board(fen)
    all_moves.update([move.uci() for move in board.legal_moves])
label_encoder = LabelEncoder()
label_encoder.fit(list(all_moves))

# Encode the target moves
df['encoded_move'] = df['target_move'].apply(lambda move: label_encoder.transform([move])[0] if move else None)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, df['encoded_move'], test_size=0.2, random_state=42)

# Building the model
model = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32)









### Calculating accuracy

import chess

def test_first_move_accuracy(X, y, model, label_encoder, df, num_puzzles=1000):
    correct_predictions = 0
    total_predictions = 0

    # Limit the number of puzzles to test
    num_puzzles = min(num_puzzles, len(X))  # Ensure we do not exceed available data

    for i in range(num_puzzles):
        board = chess.Board(df.iloc[i]['FEN'])  # Start from the initial FEN
        first_move_uci = df.iloc[i]['Moves'].split()[0]  # Opponent's first move
        
        # Apply the first move
        try:
            first_move = chess.Move.from_uci(first_move_uci)
            if first_move in board.legal_moves:
                board.push(first_move)
            else:
                continue  # skip if the first move is not legal (shouldn't happen if data is clean)
        except:
            continue  # skip on error in move conversion or pushing

        # Use the model to predict the next move
        input_tensor = parse_fen(board.fen()).reshape(1, 8, 8, 1)
        predictions = model.predict(input_tensor)
        predicted_move_idx = np.argmax(predictions[0])
        predicted_move = label_encoder.inverse_transform([predicted_move_idx])[0]
        move = chess.Move.from_uci(predicted_move)

        # Check if the predicted move matches the actual second move
        actual_second_move = df.iloc[i]['Moves'].split()[1] if len(df.iloc[i]['Moves'].split()) > 1 else None
        if predicted_move == actual_second_move:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

# Calculate accuracy
accuracy = test_first_move_accuracy(X_test, y_test, model, label_encoder, df, num_puzzles=1000)
print("First Move prediction accuracy for the first 1000 puzzles: {:.2%}".format(accuracy))





def evaluate_piece_and_move_predictions(X, y, model, label_encoder, df, num_puzzles=1000):
    correct_piece_predictions = 0
    correct_move_predictions_given_piece = 0
    total_evaluated_puzzles = 0

    num_puzzles = min(num_puzzles, len(X))  # Limit to available data
    print(f"Evaluating up to {num_puzzles} puzzles...")

    for i in range(num_puzzles):
        initial_fen = df.iloc[i]['FEN']
        moves = df.iloc[i]['Moves'].split()
        print(f"Processing puzzle {i+1}: Initial FEN = {initial_fen}, Moves = {moves}")

        if len(moves) < 2:
            print("Not enough moves to evaluate.")
            continue  # Ensure there is at least a second move to analyze

        board = chess.Board(initial_fen)
        first_move_uci = moves[0]

        try:
            first_move = chess.Move.from_uci(first_move_uci)
            if first_move in board.legal_moves:
                board.push(first_move)
            else:
                print(f"First move {first_move_uci} is not legal.")
                continue
        except chess.InvalidMoveError:
            print(f"Invalid first move UCI: {first_move_uci}")
            continue

        input_tensor = parse_fen(board.fen()).reshape(1, 8, 8, 1)
        predictions = model.predict(input_tensor)
        predicted_move_idx = np.argmax(predictions[0])
        predicted_move_uci = label_encoder.inverse_transform([predicted_move_idx])[0]

        if len(predicted_move_uci) not in [4, 5]:
            print(f"Invalid predicted UCI format: {predicted_move_uci}")
            continue

        try:
            predicted_move = chess.Move.from_uci(predicted_move_uci)
            if predicted_move not in board.legal_moves:
                print(f"Predicted move {predicted_move_uci} is not legal.")
                continue
        except chess.InvalidMoveError:
            print(f"Invalid predicted move UCI: {predicted_move_uci}")
            continue

        actual_second_move_uci = moves[1]
        try:
            actual_second_move = chess.Move.from_uci(actual_second_move_uci)
            if actual_second_move not in board.legal_moves:
                print(f"Actual second move {actual_second_move_uci} is not legal.")
                continue
        except chess.InvalidMoveError:
            print(f"Invalid second move UCI: {actual_second_move_uci}")
            continue

        predicted_piece = board.piece_at(predicted_move.from_square)
        actual_piece = board.piece_at(actual_second_move.from_square)

        if predicted_piece == actual_piece:
            correct_piece_predictions += 1
            if predicted_move == actual_second_move:
                correct_move_predictions_given_piece += 1

        total_evaluated_puzzles += 1

    print(f"Total puzzles evaluated: {total_evaluated_puzzles}")
    if total_evaluated_puzzles > 0:
        piece_prediction_accuracy = correct_piece_predictions / total_evaluated_puzzles
        move_accuracy_given_correct_piece = (correct_move_predictions_given_piece / correct_piece_predictions 
                                             if correct_piece_predictions > 0 else 0)

        print(f"Correct piece prediction accuracy: {piece_prediction_accuracy * 100:.2f}%")
        if correct_piece_predictions > 0:
            print(f"Correct move prediction accuracy given correct piece: {move_accuracy_given_correct_piece * 100:.2f}%")
    else:
        print("No puzzles were evaluated.")

# Example usage:
evaluate_piece_and_move_predictions(X_test, y_test, model, label_encoder, df, num_puzzles=1000)


print("Label Encoder Classes:", label_encoder.classes_)
