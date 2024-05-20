import numpy as np
import pandas as pd
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
# df = df.head(30000)

# Update board with a move
def update_board(fen, move):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
    return board.fen()

# Parse FEN into a numeric matrix for CNN input
def parse_fen(fen):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, 'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
    board = np.zeros((8, 8), dtype=int)
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)  # Move over empty squares
            else:
                board[i, col] = pieces[char]
                col += 1
    return board

# Extract the piece to move based on the move
def get_piece_to_move(fen, move):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    piece = board.piece_at(move.from_square)
    return piece.symbol() if piece else None

# Prepare initial data for piece prediction
def prepare_data_for_piece_prediction(df):
    df['FEN_after_first_move'] = df.apply(lambda row: update_board(row['FEN'], row['Moves'].split()[0]), axis=1)
    df['piece_to_move'] = df.apply(lambda row: get_piece_to_move(row['FEN_after_first_move'], row['Moves'].split()[1]) if len(row['Moves'].split()) > 1 else None, axis=1)
    df['board_matrix'] = df['FEN_after_first_move'].apply(parse_fen)
    return df

df = prepare_data_for_piece_prediction(df)

# Label encoding pieces and split data for the piece prediction model
label_encoder_pieces = LabelEncoder()
df['encoded_piece'] = label_encoder_pieces.fit_transform(df['piece_to_move'].dropna())
X = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))
y = df['encoded_piece']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model to predict the piece
model_piece = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_pieces.classes_), activation='softmax')
])
model_piece.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_piece.fit(X_train, y_train, epochs=10, batch_size=16)

# Prediction function for piece
def predict_piece(fen):
    board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)
    piece_pred = model_piece.predict(board_matrix)
    predicted_piece_idx = np.argmax(piece_pred)
    predicted_piece = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]
    return predicted_piece

# Evaluate predictions for both first and second moves
def evaluate_piece_predictions(df, model_piece, label_encoder_pieces, test_indices):
    correct_first_piece_predictions = 0
    correct_second_piece_predictions = 0
    total_predictions = 0
    
    for index in tqdm(test_indices, desc="Evaluating piece predictions"):
        row = df.loc[index]
        initial_fen = row['FEN']
        target_moves = row['Moves'].split()

        # Apply opponent's first move
        fen_after_opponent_first_move = update_board(initial_fen, target_moves[0])

        # Predict the piece for our first move (second move in the sequence)
        target_first_piece = row['piece_to_move']
        predicted_first_piece = predict_piece(fen_after_opponent_first_move)

        if predicted_first_piece == target_first_piece:
            correct_first_piece_predictions += 1

        # Apply our predicted first move to get the new board state
        if len(target_moves) > 1:
            fen_after_first_move = update_board(fen_after_opponent_first_move, target_moves[1])

            # Apply opponent's second move (third move in the sequence)
            fen_after_opponent_second_move = update_board(fen_after_first_move, target_moves[2])

            # Predict the piece for our second move (fourth move in the sequence)
            target_second_piece = get_piece_to_move(fen_after_opponent_second_move, target_moves[3])
            predicted_second_piece = predict_piece(fen_after_opponent_second_move)

            if predicted_second_piece == target_second_piece:
                correct_second_piece_predictions += 1

        total_predictions += 1

    first_piece_accuracy = correct_first_piece_predictions / total_predictions
    second_piece_accuracy = correct_second_piece_predictions / correct_first_piece_predictions
    total_accuracy = correct_second_piece_predictions / total_predictions

    return first_piece_accuracy, second_piece_accuracy, total_accuracy

# Running evaluation on the test set
piece_test_indices = y_test.index.tolist()  # Ensure this corresponds to the original dataframe indices
first_piece_accuracy, second_piece_accuracy, total_accuracy = evaluate_piece_predictions(df, model_piece, label_encoder_pieces, piece_test_indices)

print(f"First piece prediction accuracy: {first_piece_accuracy * 100:.2f}%")
print(f"Second piece prediction accuracy: {second_piece_accuracy * 100:.2f}%")
print(f"Total  prediction accuracy: {total_accuracy * 100:.2f}%")
