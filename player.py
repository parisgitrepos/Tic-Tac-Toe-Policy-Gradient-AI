from tic_tac_toe import TicTacToe


class Player:
    def __init__(self, team="X"):
        self.team = team

    def take_turn(self, board: TicTacToe):
        move = None
        while move not in board.valid_moves():
            board.print()
            move = input("Choose a move: ")
        return move

