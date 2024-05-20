# Imports
import chess
from tqdm import tqdm
import multiprocessing

# Import df
from LoadingData import get_df

# Data Exploration
df = get_df()
df.columns

contains_mate_in_2 = df['Themes'].str.contains('mateIn2', na=False) # Filter for mate in 2 puzzles
contains_backrank_mate = df['Themes'].str.contains('backRankMate', na=False) # Filter for backRankmate

m2_and_backrank = df[contains_mate_in_2 & contains_backrank_mate].copy() # Mate in 2 and backrank mate: length = 69583

print(df['Moves'])
print(f"Number of all puzzles: {len(df)}\n", 
      f"Number of Mate-in-2 puzzles: {len(df[contains_mate_in_2])}\n", 
      f"Number of BackRankCheckmate: {len(df[contains_backrank_mate])}\n", 
      f"Number of Mate-in-2 and BackRankCheckmate: {len(m2_and_backrank)}", sep="")

len(df['Themes'].str.contains('mateIn2', na=False))

# Function to convert a series of UCI moves to SAN moves given a starting FEN
def convert_uci_to_san(fen, uci_moves_str):
    board = chess.Board(fen)
    san_moves = []
    for uci_move in uci_moves_str.split():
        move = chess.Move.from_uci(uci_move)
        if move in board.legal_moves:
            san_move = board.san(move)
            san_moves.append(san_move)
            board.push(move)
        else:
            # If a move isn't legal, return a placeholder or handle it as needed
            san_moves.append('illegal move')
    return ' '.join(san_moves)

# Load the dataframe
df = get_df()

len(m2_and_backrank)

print(m2_and_backrank[["FEN"]])

# Use tqdm to show progress as we convert the UCI moves to SAN for each puzzle
san_moves = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting UCI to SAN"):
    san_move = convert_uci_to_san(row['FEN'], row['Moves'])
    san_moves.append(san_move)

# Add the SAN moves as a new column to the DataFrame
m2_and_backrank['SAN'] = san_moves

# Now 'mate2df' has all the original columns, plus a 'SAN' column with the SAN notation moves
print(m2_and_backrank[['Moves', 'SAN']])
 
