from tic_tac_toe import TicTacToe
from player import Player
import random
from tensorflow import keras
import numpy as np

class Q_AI(Player):

    def __init__(self, team = 'X'):
        super().__init__(team)
        self.model = keras.Sequential()

    def take_turn(self, board: TicTacToe):
        return random.choice(board.valid_moves())