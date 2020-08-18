from gym.core import Env
from tic_tac_toe import TicTacToe
import numpy as np
from player import Player
from gym.spaces import Discrete, Box


class TicTacToeEnv(Env):
    action_space = Discrete(9)
    observation_space = Box(low = 0, high = 1, shape = (3, 3, 3))

    def __init__(self, opponent_agent: Player, player = 'X'):
        self.board = TicTacToe()
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'
        self.opponent_agent = opponent_agent

    def _process_board_observation(self):
        grid = np.zeros((3, 3, 3), np.float32)
        for row in range(3):
            for column in range(3):
                if self.board.grid[row][column] == self.player:
                    grid[row][column][0] = 1
                elif self.board.grid[row][column] == self.opponent:
                    grid[row][column][1] = 1
                else:
                    grid[row][column][2] = 1
        return grid

    def step(self, action):
        moves = self.board.valid_moves()
        if not isinstance(action, int) or action < 0 or action > 8:
            raise ValueError('Action must be an integer from 0 to 8')
        move = str(action + 1)
        if move not in moves:
            raise ValueError(f'Move has already been taken, valid moves are: {moves}')
        self.board.apply_move(move, self.player)
        moves = self.board.valid_moves()
        winner = self.board.check_winner()
        done = len(moves) == 0 or winner is not None
        if not done:
            move = self.opponent_agent.take_turn(self.board)
            self.board.apply_move(move, self.opponent)
            moves = self.board.valid_moves()
            winner = self.board.check_winner()
            done = len(moves) == 0 or winner is not None
        # observation is a 3 x 3 x 3 array, first grid represents moves that we have taken,
        # second grid represents moves opponent has taken, and third grid represents
        # spaces that are open
        observation = self._process_board_observation()
        reward = 1 if winner == self.player else (-1 if winner == self.opponent else 0)
        return observation, reward, done, dict()

    def reset(self):
        self.board = TicTacToe()
        return self._process_board_observation()

    def render(self, mode='human'):
        self.board.print()
