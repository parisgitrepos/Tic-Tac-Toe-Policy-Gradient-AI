from tqdm import tqdm
from scipy.special import softmax
from tensorflow import keras
from Policy_Gradient.memory import Memory
import numpy as np
import tensorflow as tf
import mitdeeplearning as mdl
import gym
import sys


class Agent:
    def __init__(self, model: keras.Model, environment: gym.Env):
        self.model = model
        self.memory = Memory()
        self.env = environment
        self.n_actions = 9

    def choose_action(self, observation) -> int:
        """
        Function that takes observations as input, executes a forward pass through model, and outputs a sampled action.

        :param observation: observation which is fed as input to the model
        :return: choice of agent action
        """

        # add batch dimension to the observation

        # print(observation)

        free_spaces = observation[:,:,2] == 1
        observation = np.expand_dims(observation[0], axis=0)

        logits = self.model.predict(observation)

        probability = np.zeros(logits.shape, dtype = np.float32)
        probability[free_spaces] = softmax(logits[free_spaces])
        action = int(np.random.choice(self.n_actions, size=1, p=probability.flatten())[0])

        return action

    @staticmethod
    def normalize(x: np.array) -> np.array:
        """
        Helper function that normalizes an np.array x
        :param x: array of rewards to be normalized
        :return: Numpy array
        """
        x -= np.mean(x)
        x /= np.std(x)
        return x.astype(np.float32)

    def discount_rewards(self, rewards: np.array = None, gamma=0.95) -> np.array:
        """
        Compute normalized, discounted, cumulative rewards (i.e., return)
        :param rewards: reward at timesteps in episode
        :param gamma: discounting factor
        :return: normalized discounted reward
        """
        if rewards is None:
            rewards = self.memory.rewards

        discounted_rewards = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(0, len(rewards))):
            # update the total discounted reward
            R = R * gamma + rewards[t]
            discounted_rewards[t] = R

        return Agent.normalize(discounted_rewards)

    def compute_loss(self, logits, actions, rewards) -> tf.Tensor:
        """
        Computes loss of model
        :param logits: network's predictions for actions to take
        :param actions: the actions the agent took in an episode
        :param rewards: the rewards the agent received in an episode
        :return: loss
        """
        neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)

        loss = tf.reduce_mean(neg_logprob * rewards)
        return loss

    def train_step(self, optimizer: keras.optimizers.Optimizer) -> None:
        """
        Apply backpropogation and optimization to the model's trainable variables for the given episodes in memory
        :param optimizer: Tensorflow or Keras optimizer
        :return: None
        """

        with tf.GradientTape() as tape:
            # Forward propagate through the agent networks
            logits = self.model(np.vstack(self.memory.observations))

            loss = self.compute_loss(logits, np.array(self.memory.actions), self.discount_rewards(self.memory.rewards))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    def train_loop(self, optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(1e-3), episodes: int = 500,
                   model_file: str = 'model.h5', save_freq: int = 50) -> None:
        """
        Train the model for a specified number of episodes with the given optimizer
        :param optimizer: Tensorflow or Keras optimizer
        :param episodes: Number of episodes to train on
        :param model_file: File name to save model weights
        :param save_freq: Frequency of model saving
        :return: None
        """""

        # to track our progress
        smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
        plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')

        if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
        for i_episode in range(episodes):

            plotter.plot(smoothed_reward.get())

            # Restart the environment
            observation = self.env.reset()
            self.memory.clear()

            while True:
                # using our observation, choose an action and take it in the environment
                action = self.choose_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                # add to memory
                self.memory.add_to_memory(observation, action, reward)

                # is the episode over? did you crash or do so well that you're done?
                if done:
                    # determine total reward and keep a record of this
                    total_reward = sum(self.memory.rewards)
                    smoothed_reward.append(total_reward)

                    # initiate training - remember we don't know anything about how the
                    #   agent is doing until it has crashed!
                    self.train_step(optimizer)

                    # reset the memory
                    self.memory.clear()
                    break
                # update our observatons
                observation = next_observation

            if i_episode % save_freq == 0:
                self.model.save_weights(model_file)

        self.model.save_weights(model_file)