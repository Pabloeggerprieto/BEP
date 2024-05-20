from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from Data.LoadingData import DataLoader  # Ensure this import path is correct

def evaluate_model():
    model = load_model('Models/piece_model.h5')
    data_loader = DataLoader('Data/lichess_db_puzzle.csv')
    df = data_loader.get_full_dataframe()
    X_test = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))
    y_test = df['piece_to_move'].values

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == '__main__':
    evaluate_model()
