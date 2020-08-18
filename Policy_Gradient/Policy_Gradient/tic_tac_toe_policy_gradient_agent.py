from Policy_Gradient.agent import Agent
from tic_tac_toe_env import TicTacToeEnv
from ai import Min_Max_AI
import tensorflow as tf
import os


class TicTacToeAgent:

    def __init__(self, file_name = 'model.h5'):
        self.file_name = f'model/{file_name}'

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(input_shape = TicTacToeEnv.observation_space.shape, activation = 'relu', kernel_size = 2, strides = 1, filters = 32),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units = 9)
        ])

    def train(self, opponent = Min_Max_AI(team = 'O'), episodes = 500):
        env = TicTacToeEnv(opponent)

        os.makedirs('model', exist_ok = True)


        if os.path.isfile(self.file_name):
            self.model.load_weights(self.file_name)

        ticTacToe_agent = Agent(self.model, env)
        ticTacToe_agent.train_loop(model_file = self.file_name, episodes = episodes)

t = TicTacToeAgent()
t.train()