import events as e
import numpy as np
from typing import List


alpha = 0.1
gamma = 0.9
epsilon = 1.0
num_actions = 6


def setup_training(self):
    """ Initialize the training process for the agent. """
    self.q_table = {}
    self.total_rewards = []
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.num_actions = num_actions


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """ Handle learning based on the events that occurred during the game. """
    
    action_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'BOMB': 4, 'WAIT': 5}
    
    if isinstance(self_action, str):
        self_action = action_mapping.get(self_action)
    
    state_key = str(old_game_state)
    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(self.num_actions)
    
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 50
    if e.KILLED_OPPONENT in events:
        reward += 20
    if e.MOVED_UP in events or e.MOVED_DOWN in events or e.MOVED_LEFT in events or e.MOVED_RIGHT in events:
        reward += 10
    if e.BOMB_DROPPED in events:
        reward -= 5
    if e.GOT_KILLED in events:
        reward -= 20
    if e.CRATE_DESTROYED in events:
        reward += 10
    
    next_action = get_next_action(self, new_game_state)
    next_state_key = str(new_game_state)
    
    if next_state_key not in self.q_table:
        self.q_table[next_state_key] = np.zeros(self.num_actions)
    
    next_action_idx = action_mapping[next_action]
    # SARSA Update value
    old_q_value = self.q_table[state_key][self_action]
    next_q_value = self.q_table[next_state_key][next_action_idx]
    self.q_table[state_key][self_action] += self.alpha * (reward + self.gamma * next_q_value - old_q_value)


def get_next_action(self, game_state: dict):
    """ Epsilon-greedy strategy to select the next action. """
    state_key = str(game_state)
    
    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(self.num_actions)
    
    if np.random.rand() < self.epsilon:
        return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    else:
        return ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(self.q_table[state_key])]


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """ This function is called at the end of each game to reset any persistent data. """
    
    action_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'BOMB': 4, 'WAIT': 5}
    
    if isinstance(last_action, str):
        last_action = action_mapping.get(last_action)
    
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 50
    if e.KILLED_OPPONENT in events:
        reward += 20
    if e.GOT_KILLED in events:
        reward -= 20
    
    state_key = str(last_game_state)
    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(self.num_actions)
    
    self.q_table[state_key][last_action] += self.alpha * (reward - self.q_table[state_key][last_action])

    self.epsilon = max(0.1, self.epsilon * 0.99)
