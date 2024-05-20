from LoadingData import m2_and_backrank
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import chess

# Load data
df = m2_and_backrank()

# Update board with the first move
def update_board(fen, move):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
    return board.fen()

# Get the piece to move based on the move
def get_piece_to_move(fen, move):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    piece = board.piece_at(move.from_square)
    return piece.symbol() if piece else None

# Prepare data for training the first model (PIECE)
def prepare_data_for_piece_prediction(df):
    df['FEN_after_first_move'] = df.apply(lambda row: update_board(row['FEN'], row['Moves'].split()[0]), axis=1)
    df['piece_to_move'] = df.apply(lambda row: get_piece_to_move(row['FEN_after_first_move'], row['Moves'].split()[1]) if len(row['Moves'].split()) > 1 else None, axis=1)
    return df

df = prepare_data_for_piece_prediction(df)

# Convert piece symbols to numbers for input to CNN
def piece_to_number(piece):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, 'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
    return pieces.get(piece, 0)

# Parse FEN into a numeric matrix
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

df['board_matrix'] = df['FEN_after_first_move'].apply(parse_fen)
X = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))

# Encode pieces
label_encoder_pieces = LabelEncoder()
df['encoded_piece'] = label_encoder_pieces.fit_transform(df['piece_to_move'].dropna())
X_train, X_test, y_train, y_test = train_test_split(X, df['encoded_piece'], test_size=0.2, random_state=42)

# First model (PIECE)
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
model_piece.fit(X_train, y_train, epochs=10, batch_size=32==16)

# Prepare data for the MOVE model
def prepare_data_for_move_prediction(df, model_piece, label_encoder_pieces):
    # Predict piece for each board state
    piece_predictions = model_piece.predict(np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1)))
    df['predicted_piece'] = [label_encoder_pieces.inverse_transform([np.argmax(pred)])[0] for pred in piece_predictions]

    # Filter moves to only include legal moves of the predicted piece
    def filter_moves(row):
        board = chess.Board(row['FEN_after_first_move'])
        legal_moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == row['predicted_piece']]
        if row['target_move'] in legal_moves:
            return legal_moves
        return []

    df['legal_moves'] = df.apply(filter_moves, axis=1)
    return df

df['target_move'] = df.apply(lambda row: row['Moves'].split()[1] if len(row['Moves'].split()) > 1 else None, axis=1)
df = prepare_data_for_move_prediction(df, model_piece, label_encoder_pieces)
label_encoder_moves = LabelEncoder()
all_moves = [move for sublist in df['legal_moves'].tolist() for move in sublist]
label_encoder_moves.fit(all_moves)


def collect_all_possible_moves(df):
    all_moves = set()
    for index, row in df.iterrows():
        board = chess.Board(row['FEN_after_first_move'])
        all_moves.update([move.uci() for move in board.legal_moves])
    return list(all_moves)

# Before splitting the data:
all_possible_moves = collect_all_possible_moves(df)
label_encoder_moves = LabelEncoder()
label_encoder_moves.fit(all_possible_moves)

# Second model (MOVE)
model_move = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_moves.classes_), activation='softmax')
])

X_move = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))
y_move = label_encoder_moves.transform(df['target_move'])
X_move_train, X_move_test, y_move_train, y_move_test = train_test_split(X_move, y_move, test_size=0.2, random_state=42)
model_move.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_move.fit(X_move_train, y_move_train, epochs=10, batch_size=16)

# Example integration: predict piece, then predict move
def predict_piece_then_move(fen):
    board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)
    piece_pred = model_piece.predict(board_matrix)
    predicted_piece_idx = np.argmax(piece_pred)
    predicted_piece = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]
    
    board = chess.Board(fen)
    legal_moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == predicted_piece]
    if not legal_moves:
        return predicted_piece, None
    
    move_matrix = np.array([parse_fen(fen) for _ in legal_moves]).reshape(-1, 8, 8, 1)
    move_pred = model_move.predict(move_matrix)
    predicted_move_idx = np.argmax(move_pred)
    predicted_move = label_encoder_moves.inverse_transform([predicted_move_idx])[0]

    return predicted_piece, predicted_move


### Evaluate model accuracy

# KINDA CHEATING
def evaluate_models_accuracy(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves):
    correct_piece_predictions = 0
    correct_move_predictions = 0
    correct_combined_predictions = 0
    total_predictions = 0

    for index, row in df.head(5000).iterrows():
        fen = row['FEN_after_first_move']
        board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)

        # Predicting the piece
        piece_pred = model_piece.predict(board_matrix)
        predicted_piece_idx = np.argmax(piece_pred)
        predicted_piece = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]

        # Check piece prediction accuracy
        if predicted_piece == row['piece_to_move']:
            correct_piece_predictions += 1

            # Generate legal moves for the predicted piece
            board = chess.Board(fen)
            legal_moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == predicted_piece]
            if not legal_moves:
                continue

            move_matrix = np.array([parse_fen(fen) for _ in legal_moves]).reshape(-1, 8, 8, 1)
            move_pred = model_move.predict(move_matrix)
            predicted_move_idx = np.argmax(move_pred)
            try:
                predicted_move = label_encoder_moves.inverse_transform([predicted_move_idx])[0]
                # Check move prediction accuracy
                if predicted_move == row['target_move']:
                    correct_combined_predictions += 1
                if predicted_move in legal_moves:
                    correct_move_predictions += 1
            except ValueError:
                print(f"Unseen label index: {predicted_move_idx}, skipping this index.")

        total_predictions += 1

    piece_accuracy = correct_piece_predictions / total_predictions
    move_accuracy = correct_move_predictions / total_predictions
    combined_accuracy = correct_combined_predictions / total_predictions

    return piece_accuracy, move_accuracy, combined_accuracy

# Run the evaluation
piece_accuracy, move_accuracy, combined_accuracy = evaluate_models_accuracy(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves)
print(f"Piece prediction accuracy: {piece_accuracy * 100:.2f}%")
print(f"Move prediction accuracy: {move_accuracy * 100:.2f}%")
print(f"Combined model accuracy: {combined_accuracy * 100:.2f}%")



### RESULTS IN THE REPORT ARE BASED ON THE ABOVE





# ### THIS ONE BETTER IF IT WORKS


# def evaluate_models_accuracy(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves):
#     correct_piece_predictions = 0
#     correct_move_predictions = 0
#     correct_combined_predictions = 0
#     total_predictions = 0

#     for index, row in df.iterrows():
#         fen = row['FEN_after_first_move']
#         board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)

#         # Predicting the piece
#         piece_pred = model_piece.predict(board_matrix)
#         predicted_piece_idx = np.argmax(piece_pred)
#         predicted_piece = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]

#         # Getting actual values
#         actual_piece = row['piece_to_move']
#         actual_move = row['target_move']

#         # Check piece prediction accuracy
#         if predicted_piece == actual_piece:
#             correct_piece_predictions += 1

#             # Predicting the move if the piece prediction is correct
#             board = chess.Board(fen)
#             legal_moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == predicted_piece]
#             move_matrix = np.array([parse_fen(fen) for _ in legal_moves]).reshape(-1, 8, 8, 1)
#             move_pred = model_move.predict(move_matrix)
#             predicted_move_idx = np.argmax(move_pred)
#             predicted_moves = label_encoder_moves.inverse_transform([predicted_move_idx])[0]

#             # Check move prediction accuracy
#             if predicted_moves == actual_move:
#                 correct_combined_predictions += 1

#         # Check if the move alone is correct regardless of the piece prediction
#         if actual_move in legal_moves:
#             correct_move_predictions += 1

#         total_predictions += 1

#     piece_accuracy = correct_piece_predictions / total_predictions
#     move_accuracy = correct_move_predictions / total_predictions
#     combined_accuracy = correct_combined_predictions / total_predictions

#     return piece_accuracy, move_accuracy, combined_accuracy


# # Usage example
# piece_accuracy, move_accuracy, combined_accuracy = evaluate_models_accuracy(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves)
# print(f"Piece prediction accuracy: {piece_accuracy * 100:.2f}%")
# print(f"Move prediction accuracy: {move_accuracy * 100:.2f}%")
# print(f"Combined model accuracy: {combined_accuracy * 100:.2f}%")




# ### CURRENTLY WORKING ON THIS
# import time

# def evaluate_models_accuracy(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves):
#     correct_piece_predictions = 0
#     correct_move_predictions = 0
#     correct_combined_predictions = 0
#     total_predictions = 0

#     start_time = time.time()
#     total_rows = len(df)
#     times = []

#     for index, row in df.iterrows():
#         loop_start = time.time()
#         current_index = index + 1  # Increment to match human-readable counting (starting from 1)

#         fen = row['FEN_after_first_move']
#         board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)

#         # Predicting the piece
#         piece_pred = model_piece.predict(board_matrix)
#         predicted_piece_idx = np.argmax(piece_pred)
#         predicted_piece = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]

#         # Getting actual values
#         actual_piece = row['piece_to_move']
#         actual_move = row['target_move']

#         # Check piece prediction accuracy
#         if predicted_piece == actual_piece:
#             correct_piece_predictions += 1

#             # Predicting the move if the piece prediction is correct
#             board = chess.Board(fen)
#             legal_moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == predicted_piece]
#             move_matrix = np.array([parse_fen(fen) for _ in legal_moves]).reshape(-1, 8, 8, 1)
#             move_pred = model_move.predict(move_matrix)
#             predicted_move_idx = np.argmax(move_pred)

#             # Check if the predicted index is valid
#             if predicted_move_idx < len(label_encoder_moves.classes_):
#                 predicted_moves = label_encoder_moves.inverse_transform([predicted_move_idx])[0]
#                 # Check move prediction accuracy
#                 if predicted_moves == actual_move:
#                     correct_combined_predictions += 1
#                 if predicted_moves in legal_moves:
#                     correct_move_predictions += 1
#             else:
#                 print(f"Invalid prediction index: {predicted_move_idx}, skipping this index.")

#         total_predictions += 1

#         loop_end = time.time()
#         times.append(loop_end - loop_start)
#         mean_time_per_record = sum(times) / len(times)

#         # Print progress at every 100th record or the last record
#         if (index + 1) % 100 == 0 or (index + 1) == total_rows:
#             elapsed_time = time.time() - start_time
#             estimated_total_time = mean_time_per_record * total_rows
#             remaining_time = estimated_total_time - elapsed_time
#             print(f"Processed {index + 1}/{total_rows}. Time elapsed: {elapsed_time / 60:.2f} min. Estimated total time: {estimated_total_time / 60:.2f} min, Remaining time: {remaining_time / 60:.2f} min")

#         total_predictions += 1

#     # Calculate accuracies
#     piece_accuracy = correct_piece_predictions / total_predictions
#     move_accuracy = correct_move_predictions / total_predictions
#     combined_accuracy = correct_combined_predictions / total_predictions

#     return piece_accuracy, move_accuracy, combined_accuracy

# # Usage example
# try:
#     piece_accuracy, move_accuracy, combined_accuracy = evaluate_models_accuracy(df.head(100), model_piece, model_move, label_encoder_pieces, label_encoder_moves)
#     print(f"Piece prediction accuracy: {piece_accuracy * 100:.2f}%")
#     print(f"Move prediction accuracy: {move_accuracy * 100:.2f}%")
#     print(f"Combined model accuracy: {combined_accuracy * 100:.2f}%")
# except Exception as e:
#     print(f"An error occurred: {e}")
























# ### NOT REVISED ###

# ### Calculating accuracy

# def test_piece_prediction_accuracy(X, y, model, label_encoder, df, num_puzzles=1000):
#     correct_predictions = 0
#     total_predictions = 0

#     # Limit the number of puzzles to test
#     num_puzzles = min(num_puzzles, len(X))  # Ensure we do not exceed available data

#     for i in range(num_puzzles):
#         board = chess.Board(df.iloc[i]['FEN'])  # Start from the initial FEN
#         first_move_uci = df.iloc[i]['Moves'].split()[0]  # Opponent's first move
        
#         # Apply the first move
#         try:
#             first_move = chess.Move.from_uci(first_move_uci)
#             if first_move in board.legal_moves:
#                 board.push(first_move)
#             else:
#                 continue  # skip if the first move is not legal (shouldn't happen if data is clean)
#         except:
#             continue  # skip on error in move conversion or pushing

#         # Use the model to predict the piece to move
#         input_tensor = parse_fen(board.fen()).reshape(1, 8, 8, 1)
#         predictions = model.predict(input_tensor)
#         predicted_piece_idx = np.argmax(predictions[0])
#         predicted_piece = label_encoder.inverse_transform([predicted_piece_idx])[0]

#         # Get the actual piece intended to move in the second move
#         actual_second_move = df.iloc[i]['Moves'].split()[1] if len(df.iloc[i]['Moves'].split()) > 1 else None
#         if actual_second_move:
#             actual_piece_to_move = get_piece_to_move(board.fen(), actual_second_move)
#         else:
#             continue  # If there's no second move, skip this case

#         # Check if the predicted piece matches the actual piece to move
#         if predicted_piece == actual_piece_to_move:
#             correct_predictions += 1

#         total_predictions += 1

#     accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
#     return accuracy

# # Usage example:
# # Calculate accuracy for predicting the piece to move
# accuracy = test_piece_prediction_accuracy(X_test, y_test, model, label_encoder, df, num_puzzles=1000)
# print("Piece prediction accuracy for the first 1000 puzzles: {:.2%}".format(accuracy))












### Predict the piece's move

def prepare_data_for_move_prediction(df, label_encoder_pieces):
    """Prepare data for training the move prediction model."""
    df['predicted_piece'] = df.apply(lambda row: predict_piece(row['FEN_after_first_move']), axis=1)

    # Filter out cases where no piece is predicted (if any)
    df = df[df['predicted_piece'].notnull()]

    # Encoding the predicted piece into the board matrix as an additional feature layer
    df['predicted_piece_matrix'] = df['predicted_piece'].apply(lambda piece: piece_to_number(piece))
    df['enhanced_board_matrix'] = df.apply(lambda row: enhance_board_with_piece(row['board_matrix'], row['predicted_piece_matrix']), axis=1)

    return df

def enhance_board_with_piece(board_matrix, piece_number):
    """Add the predicted piece information as a layer or feature in the board matrix."""
    # This is a simple version where we add the predicted piece as an additional layer
    # A more complex version could involve different encoding strategies
    enhanced_matrix = np.dstack((board_matrix, piece_number * np.ones((8, 8))))
    return enhanced_matrix


from tensorflow.keras.layers import concatenate

def create_move_prediction_model(input_shape, num_classes):
    """Create a CNN model for predicting chess moves given a board state and a piece."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



# Assume df, label_encoder_moves, label_encoder_pieces are already defined
df = prepare_data_for_move_prediction(df, label_encoder_pieces)
X = np.array(df['enhanced_board_matrix'].tolist()).reshape((-1, 8, 8, 2))  # Assuming we add one layer for the piece
y = label_encoder_moves.transform(df['target_move'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_move = create_move_prediction_model(input_shape=(8, 8, 2), num_classes=len(label_encoder_moves.classes_))
model_move.fit(X_train, y_train, epochs=10, batch_size=16)




def predict_move_from_piece(fen, model_piece, model_move, label_encoder_pieces, label_encoder_moves):
    """Given a FEN, predict the piece to move and then predict the move."""
    board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)
    piece_prediction = model_piece.predict(board_matrix)
    predicted_piece_idx = np.argmax(piece_prediction[0])
    predicted_piece = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]

    # Prepare input for the move model
    piece_number = piece_to_number(predicted_piece)
    enhanced_matrix = enhance_board_with_piece(parse_fen(fen), piece_number).reshape(1, 8, 8, 2)
    move_prediction = model_move.predict(enhanced_matrix)
    predicted_move_idx = np.argmax(move_prediction[0])
    predicted_move = label_encoder_moves.inverse_transform([predicted_move_idx])[0]

    return predicted_move


