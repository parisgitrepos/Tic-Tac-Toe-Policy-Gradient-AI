class Memory:
    def __init__(self):
        self.clear()

    def clear(self) -> None:
        """
        Resets/restarts the memory buffer.
        :return: None
        """
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward) -> None:
        """
        Add observations, actions, rewards to memory.
        :param new_observation: Next observation at a new time step.
        :param new_action: Next action at a new time step.
        :param new_reward: Next reward at a new time step.
        :return: None
        """
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)