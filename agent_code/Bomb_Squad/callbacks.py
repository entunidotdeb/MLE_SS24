import numpy as np

def setup(self):
    """ Called once before the first game starts to initialize agent. """
    self.current_round = 0
    self.epsilon = 1.0  
    self.num_actions = 6  
    self.q_table = {}  
    self.alpha = 0.1  
    self.gamma = 0.9  

def act(self, game_state: dict):
    """ Called every step to determine the next action. """
    
    state_key = str(game_state)
    
    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(self.num_actions)
    
    if np.random.rand() < self.epsilon:
        return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    else:
        q_values = self.q_table[state_key]
        return ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(q_values)]
    
    if self.current_round > 50:
        self.epsilon = max(0.1, self.epsilon * 0.99)

def reward_update(self, events, old_game_state, self_action, new_game_state):
    """ Update the Q-table with the SARSA update rule. """
    
    action_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'BOMB': 4, 'WAIT': 5}
    self_action_idx = action_mapping[self_action]
    
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 50
    
    next_action = self.act(new_game_state)
    next_action_idx = action_mapping[next_action]
    # SARSA update rule
    old_q_value = self.q_table[str(old_game_state)][self_action_idx]
    next_q_value = self.q_table[str(new_game_state)][next_action_idx]
    
    self.q_table[str(old_game_state)][self_action_idx] += self.alpha * (reward + self.gamma * next_q_value - old_q_value)

def end_of_game(self, events):
    """ Reset persistent data after each game. """
    self.current_round += 1
