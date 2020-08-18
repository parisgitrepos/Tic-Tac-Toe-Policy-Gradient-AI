import copy
# We want to make an AI that plays TicTacToe intelligently
# In order for our AI to perform intelligently,
# we need our AI to understand what the game board represents -> define our AI state (inputs)
# We need to define a class/structure for our TicTacToe board
class TicTacToe:
    # Whenever a class fuction is created, pass self through first
    def __init__(self):
        # how can we represent our 3x3 grid?
        self.grid = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    # Tic Tac Toe will have a grid and once we call Tic Tac Toe and choose to print, it will print the tic tac toe grid
    # Any function that we have can modify self and can access the project

    # define a function print
    def print(self):
        for row in self.grid:
            print('|'.join(row)) # print out all the items in our row with | between them 1|2|3

    def valid_moves(self):
        # assume there are no valid moves (empty list)
        moves = []

        # check every row
        for row in self.grid:
            # check every cell in every row
            for cell in row:
                # check if the cell is not an X or O
                if cell.upper() not in ["X", "O"]:
                    # the cell is not X or O, so it is a valid move
                    # add this move to our list of valid moves
                    moves.append(cell)
        # After checking every cell, return any valid moves we found previously
        return moves

    def apply_move(self, move, player = "X"):
        # Make sure the move is valid first
        if move in self.valid_moves():
            # since the move is valid, we need to find the cell which needs to be replaced
            # check every row number
            for row_number in range(3):
                # check every column number for every row number
                for column_number in range(3):
                    # if this cell has the move we requested,
                    if move == self.grid[row_number][column_number]:
                        # replace this tile in our grid with whichever player made this move
                        self.grid[row_number][column_number] = player
        return self

    def check_winner(self):
        # check each row
        for row in self.grid:
            # check if the first, second, and third items (all of them) are the same
            if row[0] == row[1] and row[0] == row[2]:
                # someone has 3 in a row, whoever has this row won
                return row[0]
        # check each column number
        for column_number in range(3):
            # first item in our column (for shorter code later)
            tile = self.grid[0][column_number]
            # if the first item in our column is the same as the second and third (all items match)
            if tile == self.grid[1][column_number] and tile == self.grid[2][column_number]:
                # whoever has a tile in this column won
                return tile
        # check the diagonal lines in a similar manner
        center = self.grid[1][1]
        if center == self.grid[0][0] and center == self.grid[2][2]:
            return center

        if center == self.grid[0][2] and center == self.grid[2][0]:
            return center
# # Class -> custom structure with variables and functions inside
# game = TicTacToe()
# game.print()
# print(game.grid)
#
# # print("Valid moves", game.valid_moves())
# #
# # move = input("Choose a move: ")
# # game.apply_move(move)
# # game.print()
# # move = input("Choose a move: ")
# # game.apply_move(move, "O")
# # game.print()
#
# turn = "X"
# moves = game.valid_moves()
# while len(moves) > 0 and game.check_winner() is None:
#     game.print()
#     move = input("Choose a move: ")
#     if move in moves:
#         game.apply_move(move, turn)
#         turn = "O" if turn == "X" else "X"
# game.print()
# print(game.check_winner())