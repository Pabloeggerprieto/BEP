from LoadingData import m2_and_backrank
# from ChessRules import get_legal_moves
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import chess
import random


df = m2_and_backrank

# Encoding pieces for the chess board
def piece_to_number(piece):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
              'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
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

# Apply parse_fen to each FEN string and reshape for the model
df['board_matrix'] = df['FEN'].apply(parse_fen)
X = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))

# Get legal moves for encoding
def get_legal_moves(fen):
    board = chess.Board(fen)
    return [move.uci() for move in board.legal_moves]


# Label encode moves
all_moves = set()
for fen in df['FEN']:
    all_moves.update(get_legal_moves(fen))
label_encoder = LabelEncoder()
label_encoder.fit(list(all_moves))

# Encode the moves directly as integers
def encode_moves(fen, moves):
    board = chess.Board(fen)
    legal_moves = [move.uci() for move in board.legal_moves]
    moves = moves.split()
    legal_encoded = [label_encoder.transform([move])[0] for move in moves if move in legal_moves]
    return legal_encoded

# Apply encoding
df['encoded_moves'] = [encode_moves(fen, moves) for fen, moves in zip(df['FEN'], df['Moves'])]

max_length = max(len(moves) for moves in df['encoded_moves'])
y = np.array([moves[0] if len(moves) > 0 else -1 for moves in df['encoded_moves']])  # No padding needed


# Creating the model with Embedding layer
model = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Predict a single move per input
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32)


# Evaluate the model on the training data
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=1)

# Print the accuracy
print("Accuracy on training data: {:.2f}%".format(train_accuracy * 100))



def simulate_puzzle(board_fen, model, label_encoder):
    board = chess.Board(board_fen)
    success = True

    # Simulate the puzzle sequence (assuming two moves needed to solve the puzzle)
    for _ in range(2):  # Adjust the range if more moves are required
        input_tensor = parse_fen(board.fen()).reshape(1, 8, 8, 1)
        predictions = model.predict(input_tensor)
        predicted_move_idx = np.argmax(predictions[0])
        predicted_move_uci = label_encoder.inverse_transform([predicted_move_idx])[0]
        move = chess.Move.from_uci(predicted_move_uci)

        if move in board.legal_moves:
            board.push(move)
        else:
            success = False
            break

        # Check if the puzzle is solved
        if board.is_checkmate():
            break
        else:
            # If not solved, apply the next move in the puzzle (this would typically be your opponent's move)
            # Here, it's assumed the next move is needed and known, this would be part of the puzzle's move sequence
            continue

    return success

# Evaluate the model's ability to solve full puzzles
def evaluate_puzzle_solving_ability(df, model, label_encoder):
    total_puzzles = len(df)
    solved_puzzles = sum(simulate_puzzle(row['FEN'], model, label_encoder) for index, row in df.iterrows())

    print(f"Solved {solved_puzzles} out of {total_puzzles} puzzles.")
    print(f"Puzzle Solving Accuracy: {solved_puzzles / total_puzzles:.2%}")

# Call the evaluation function
evaluate_puzzle_solving_ability(df.head(1000), model, label_encoder)

































# # Calculating accuracy

# def solve_puzzle(board, model, label_encoder):
#     """ Attempts to solve a puzzle and returns True if successful. """
#     success = True
#     for _ in range(2):  # Assuming "mate in 2" puzzles
#         # Predict the next move
#         input_tensor = parse_fen(board.fen()).reshape(1, 8, 8, 1)
#         predicted_probabilities = model.predict(input_tensor)[0]
#         predicted_move_idx = np.argmax(predicted_probabilities)
#         predicted_move = label_encoder.inverse_transform([predicted_move_idx])[0]

#         # Check if the move is legal and make the move
#         move = chess.Move.from_uci(predicted_move)
#         if move in board.legal_moves:
#             board.push(move)
#         else:
#             success = False
#             break

#         # Check for checkmate
#         if board.is_checkmate():
#             break
#         else:
#             # Let's assume the opponent makes a random legal move
#             opponent_move = random.choice(list(board.legal_moves))
#             board.push(opponent_move)

#     return success and board.is_checkmate()

# # Test the model on all puzzles
# successful_puzzles = 0
# total_puzzles = len(df)

# for index, row in df.iterrows():
#     board = chess.Board(row['FEN'])
#     if solve_puzzle(board, model, label_encoder):
#         successful_puzzles += 1

# # Calculate accuracy of solving puzzles
# puzzle_solving_accuracy = successful_puzzles / total_puzzles
# print("Puzzle Solving Accuracy: {:.2%}".format(puzzle_solving_accuracy))
