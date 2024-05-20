import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Import the DataLoader class from your DataLoader script
from Data.LoadingData import DataLoader

def build_model(input_shape, num_classes):
    """Build and compile the CNN model for predicting chess pieces."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # Initialize the data loader
    data_loader = DataLoader('Data/lichess_db_puzzle.csv')
    
    # Get only the Mate in 2 puzzles
    df = data_loader.get_mate_in_2_and_backrank()
    X = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))  # Ensure the matrix size matches your actual data shape
    y = df['piece_to_move'].values

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1:], len(np.unique(y_encoded)))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    # Save the trained model and the label encoder
    model_path = os.path.join('models', 'model_piece.h5')
    encoder_path = os.path.join('label_encoders', 'piece_label_encoder.pkl')
    model.save(model_path)

    # Save the encoder for future use
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

if __name__ == '__main__':
    train_model()
