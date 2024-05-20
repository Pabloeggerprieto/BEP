import chess

def get_legal_moves(fen):
    #
    # Returns all legal moves given a position
    #
    board = chess.Board(fen)
    return [move.uci() for move in board.legal_moves]

def make_move_on_board(fen, move):
    #
    # Makes the move and updates the board
    #
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
        return board.fen()
    else:
        return None  # or handle illegal move
    


