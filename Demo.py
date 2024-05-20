import chess
import random
import time
from IPython.display import display, clear_output


board = chess.Board()
move = 0

while move < 10:
    alllegalmoves = []
    for legalmove in board.legal_moves:
        alllegalmoves.append(str(legalmove))
        move_this_turn = random.choice(alllegalmoves)
    move_this_turn = chess.Move.from_uci(move_this_turn)
    board.push(move_this_turn)
    print(board, '\n')
    move += 1    

board.push(move_this_turn)
# print(move_this_turn)
print(board)

legalmove = random.choice(board.legal_moves)






# Initialize a chess board
board = chess.Board()

# Continue the game until it's checkmate or stalemate
while not board.is_game_over():
    # Generate all legal moves
    legal_moves = list(board.legal_moves)
    
    # Select a random move from the legal moves
    random_move = random.choice(legal_moves)
    
    # Make the move on the board
    board.push(random_move)
    
    # Print the board state to visualize the game
    print(board)
    print("\n")

# Check the final status of the game
if board.is_checkmate():
    print("Checkmate.")
elif board.is_stalemate():
    print("Stalemate.")
else:
    print("Game over with a different termination condition.")

# Output the final board position
print("Final board position:")
print(board)






# Initialize a chess board
board = chess.Board()

# Display the initial board
display(board)

# Continue the game until it's checkmate or stalemate
while not board.is_game_over():
    # Generate all legal moves
    legal_moves = list(board.legal_moves)
    
    # Select a random move from the legal moves
    random_move = random.choice(legal_moves)
    
    # Make the move on the board
    board.push(random_move)
    
    # Clear the previous board display and display the new board state
    clear_output(wait=True)
    display(board, '\n')
    
    # Wait for 1 second before making the next move
    time.sleep(1)

# Output the final game status
if board.is_checkmate():
    print("Checkmate.")
elif board.is_stalemate():
    print("Stalemate.")