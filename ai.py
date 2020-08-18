from tic_tac_toe import TicTacToe
from player import Player
import random
import copy

class Random_AI(Player):
    def take_turn(self, board: TicTacToe):
        return random.choice(board.valid_moves())

class Choose_First_AI (Player):
    def take_turn(self, board: TicTacToe):
        moves = board.valid_moves()
        return moves[0]



class Min_Max_AI (Player):
    def take_turn(self, board: TicTacToe):
        predicted_value, move = self.minmax(board, 6, True)
        print(self.team, "predicts that they will get a score of", predicted_value)
        return move

    def heuristic (self, board: TicTacToe):
        # Score board based on win, lose, draw
        winner = board.check_winner()
        if winner == self.team:
            return 1
        elif winner is not None:
            return -1
        else:
            return 0
    def minmax (self, board: TicTacToe, depth: int, maximizing: bool):
        moves = board.valid_moves()
        if len(moves)<1 or depth<1 or board.check_winner() is not None:
            return self.heuristic(board), None
        if maximizing:
            # We are playing for ourselves, we want the highest score
            # We've made our move assuming the opponent makes a move where we get worse and they get better
            max_value = -2 # we should find minimums less than this
            max_i = 0
            for i in range(len(moves)):
                outcome = copy.deepcopy(board).apply_move(moves[i], self.team)
                value , move = self.minmax(outcome, depth - 1, False)
                if value > max_value:
                    max_i = i
                    max_value = value
            # Our opponent chose the worst possible things for us, so we will choose the best possible option out of these
            return max_value, moves[max_i]
        else:
            # We are pretending to be the opponent
            opponent_team = 'O' if self.team == 'X' else 'X'
            # We assume the player made the best/max move it can make for highscores
            min_value = 2  # we should find minimums less than this
            min_i = 0
            for i in range(len(moves)):
                outcome = copy.deepcopy(board).apply_move(moves[i], opponent_team)
                value, move = self.minmax(outcome, depth - 1, True)
                if value < min_value:
                    min_i = i
                    min_value = value
            # Our opponent chose the worst possible things for us, so we will choose the best possible option out of these
            return min_value, moves[min_i]