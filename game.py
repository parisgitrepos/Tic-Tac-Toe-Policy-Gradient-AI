from tic_tac_toe import TicTacToe
from player import Player
from ai import Random_AI, Choose_First_AI, Min_Max_AI

class Game:
    # Sets up game
    def __init__(self, x_player: Player, o_player: Player):
        self.x_player = x_player
        self.o_player = o_player
        self.board = TicTacToe()
    # Runs the game from start to finish
    def run(self):
        current_player = self.x_player
        moves = self.board.valid_moves()
        while len(moves) > 0 and self.board.check_winner() is None:
            print("Current player:", current_player.team)
            move = current_player.take_turn(self.board)
            if move in moves:
                self.board.apply_move(move, current_player.team)
                # Swapping player turns
                if current_player == self.x_player:
                    current_player = self.o_player
                else:
                    current_player = self.x_player
                moves = self.board.valid_moves()
        self.board.print()
        print("Winner:", self.board.check_winner())


human_x = Player("X")
human_o = Player("O")
random_x = Random_AI("X")
random_o = Random_AI("O")
first_x = Choose_First_AI("X")
first_o = Choose_First_AI("O")
minmax_x = Min_Max_AI("X")
minmax_o = Min_Max_AI("O")

# print("Random AI")
# game = Game(random_x, random_o)
# game.run()
# print("Choosing First vs. Random")
# game = Game(first_x, random_o)
# game.run()
# print("Min_Max vs. Random")
# game = Game(minmax_x, random_o)
# game.run()
# print("Min_Max vs. Min_Max")
# game = Game(minmax_x, minmax_o)
# game.run()
# print("Random vs. Min_Max")
# game = Game(random_x, minmax_o)
# game.run()
# print("Minmax vs. Human")
# game = Game(minmax_x, human_o)
# game.run()
# print("Human vs Human")
# game = Game(human_x, human_o)
# game.run()
# print("Human vs Random AI")
# game = Game(human_x, random_o)
# game.run()

game = Game(minmax_x, human_o)
game.run()
