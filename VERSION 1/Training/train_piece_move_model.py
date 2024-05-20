import numpy as np
import pandas as pd
import chess
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from Data.LoadingData import DataLoader

# Initialize DataLoader with the file path
loader = DataLoader('Data/lichess_db_puzzle.csv.zst')

# Load data
df = loader.backrank_and_m2()
df = df.head(5000)

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
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_pieces.classes_), activation='softmax')
])
model_piece.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_piece.fit(X_train, y_train, epochs=10, batch_size=16)

# Collecting all possible moves made by each piece from the legal moves
def collect_legal_moves(df):
    legal_moves = set()
    for index, row in df.iterrows():
        board = chess.Board(row['FEN_after_first_move'])
        moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == row['piece_to_move']]
        legal_moves.update(moves)
    return list(legal_moves)

# Ensure all possible moves are included for encoding
all_legal_moves = collect_legal_moves(df)
label_encoder_moves = LabelEncoder()
label_encoder_moves.fit(all_legal_moves)

# Function to filter moves for a given piece
def filter_moves_for_piece(fen, piece):
    board = chess.Board(fen)
    return [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == piece]

# Prepare data for the move model
def prepare_move_data(df):
    X = []
    y = []
    for index, row in df.iterrows():
        fen = row['FEN_after_first_move']
        piece = row['piece_to_move']
        target_move = row['Moves'].split()[1] if len(row['Moves'].split()) > 1 else None

        filtered_moves = filter_moves_for_piece(fen, piece)
        if target_move in filtered_moves:
            board_matrix = parse_fen(fen)
            X.append(board_matrix)
            y.append(label_encoder_moves.transform([target_move])[0])

    X = np.array(X).reshape((-1, 8, 8, 1))
    y = to_categorical(y, num_classes=len(label_encoder_moves.classes_))
    return X, y

X_move, y_move = prepare_move_data(df)
X_train_move, X_test_move, y_train_move, y_test_move = train_test_split(X_move, y_move, test_size=0.2, random_state=42)

# Model to predict the move
model_move = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_moves.classes_), activation='softmax')
])

model_move.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_move.fit(X_train_move, y_train_move, epochs=10, batch_size=16)

# Prediction function for piece and then move
def predict_piece_then_move(fen):
    board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)
    piece_pred = model_piece.predict(board_matrix)
    predicted_piece_idx = np.argmax(piece_pred)
    predicted_piece = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]

    board = chess.Board(fen)
    legal_moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == predicted_piece]

    if not legal_moves:
        return predicted_piece, None  # No legal moves for the predicted piece

    # Filter legal moves for the predicted piece
    legal_move_matrices = np.array([parse_fen(update_board(fen, move)) for move in legal_moves]).reshape(-1, 8, 8, 1)

    # Predict the move using the filtered legal moves
    move_preds = model_move.predict(legal_move_matrices)
    predicted_move_idx = np.argmax(move_preds)
    predicted_move = legal_moves[predicted_move_idx] if len(legal_moves) > predicted_move_idx else None

    return predicted_piece, predicted_move

# Evaluate predictions for both first and second moves
def evaluate_predictions(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves, piece_test_indices, move_test_indices):
    correct_piece_predictions = 0
    correct_move_predictions = 0
    correct_second_piece_predictions = 0
    correct_second_move_predictions = 0
    total_predictions = 0

    for index in tqdm(piece_test_indices, desc="Evaluating piece predictions"):
        row = df.loc[index]
        initial_fen = row['FEN']
        target_moves = row['Moves'].split()
        
        # Apply opponent's first move
        fen_after_opponent_first_move = update_board(initial_fen, target_moves[0])

        # First move predictions (second move in the sequence)
        fen_after_first_move = fen_after_opponent_first_move
        target_first_piece = row['piece_to_move']
        target_first_move = target_moves[1] if len(target_moves) > 1 else None

        predicted_first_piece, predicted_first_move = predict_piece_then_move(fen_after_first_move)

        if predicted_first_piece == target_first_piece:
            correct_piece_predictions += 1

            if predicted_first_move == target_first_move:
                correct_move_predictions += 1

                # Update board state to reflect our first move (second in the sequence)
                fen_after_second_move = update_board(fen_after_first_move, predicted_first_move)

                # Apply opponent's second move (third in the sequence)
                fen_after_opponent_second_move = update_board(fen_after_second_move, target_moves[2])

                # Second move predictions (fourth move in the sequence)
                fen_after_third_move = fen_after_opponent_second_move
                target_second_piece = get_piece_to_move(fen_after_third_move, target_moves[3])
                predicted_second_piece, predicted_second_move = predict_piece_then_move(fen_after_third_move)

                if predicted_second_piece == target_second_piece:
                    correct_second_piece_predictions += 1

                    if predicted_second_move == target_moves[3]:
                        correct_second_move_predictions += 1

        total_predictions += 1

    piece_accuracy = correct_piece_predictions / total_predictions
    move_accuracy = correct_move_predictions / total_predictions
    second_piece_accuracy = correct_second_piece_predictions / total_predictions
    second_move_accuracy = correct_second_move_predictions / total_predictions

    return piece_accuracy, move_accuracy, second_piece_accuracy, second_move_accuracy

# Running evaluation on the test set
piece_test_indices = y_test.index.tolist()  # Ensure this corresponds to the original dataframe indices
move_test_indices = y_test_move.index.tolist()  # Ensure this corresponds to the original dataframe indices
piece_accuracy, move_accuracy, second_piece_accuracy, second_move_accuracy = evaluate_predictions(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves, piece_test_indices, move_test_indices)

print(f"First piece prediction accuracy: {piece_accuracy * 100:.2f}%")
print(f"First move prediction accuracy: {move_accuracy * 100:.2f}%")
print(f"Second piece prediction accuracy: {second_piece_accuracy * 100:.2f}%")
print(f"Second move prediction accuracy: {second_move_accuracy * 100:.2f}%")
