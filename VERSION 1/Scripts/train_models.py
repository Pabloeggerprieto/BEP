import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import chess

# Import the DataLoader class from your DataLoader script
from Data.LoadingData import DataLoader



# Example of initializing DataLoader with a correct path
data_loader = DataLoader('path/to/your/lichess_db_puzzle.csv')

# Get the entire dataset or the filtered subset
full_df = data_loader.get_full_dataframe()
mate_in_2_df = data_loader.get_mate_in_2_and_backrank()

# Example usage of the data
print(full_df.head())
print(mate_in_2_df.head())

# Define necessary functions
def update_board(fen, move):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
    return board.fen()

def parse_fen(fen):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, 'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
    board = np.zeros((8, 8), dtype=int)
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board[i, col] = pieces[char]
                col += 1
    return board

def prepare_data_for_second_move(df):
    df['FEN_after_first_move'] = df.apply(lambda row: update_board(row['FEN'], row['Moves'].split()[0]), axis=1)
    df['target_move'] = df['Moves'].apply(lambda moves: moves.split()[1] if len(moves.split()) > 1 else None)
    df['board_matrix'] = df['FEN_after_first_move'].apply(parse_fen)
    return df

# Main function to load data, train and save the model
def train_and_save_model():
    from Data.LoadingData import DataLoader

    # Load and prepare the data
    data_loader = DataLoader('Data/lichess_db_puzzle.csv')
    df = data_loader.get_mate_in_2_and_backrank()  # Assuming this method filters the dataset correctly
    df = prepare_data_for_second_move(df)

    X = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))
    y = df['target_move'].values

    # Encode the moves
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model and label encoder
    model.save('Models/model_piece.h5')
    with open('Label_Encoders/piece_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

if __name__ == '__main__':
    train_and_save_model()
