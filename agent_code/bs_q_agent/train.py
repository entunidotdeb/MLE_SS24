from collections import namedtuple, deque
import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import ACTIONS, get_state

# Define a namedtuple for storing transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 10000  # Size of the transition history
RECORD_ENEMY_TRANSITIONS = 0.0  # Not recording enemy transitions

def setup_training(self):
    """
    Initialize self for training purposes.

    This is called after `setup` in callbacks.py.
    """
    # Initialize the Q-table and learning parameters
    self.q_table = dict()
    self.alpha = 0.2  # Learning rate
    self.gamma = 0.7  # Discount factor
    self.epsilon = 1.0  # Initial exploration rate
    self.epsilon_decay = 0.9995  # Decay rate for exploration
    self.epsilon_min = 0.01  # Minimum exploration rate

    # For storing transitions
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Initialize variables to store the last state and action
    self.last_state = None
    self.last_action = None

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    Here we update the Q-table based on the transition from old_state to new_state.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Get the state and action from the last step
    state = self.last_state
    action = self.last_action

    # Get the new state and suggested action
    next_state, _ = get_state(self, new_game_state)

    # Skip if the state is None
    if state is None or action is None or next_state is None:
        return

    # Get the reward from the events
    reward = reward_from_events(self, events)

    # Store the transition
    self.transitions.append(Transition(state, action, next_state, reward))

    # Update Q-table
    update_q_table(self, state, action, next_state, reward)

    # Decay epsilon
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    Here we update the Q-table for the last transition and save the model.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # Get the state and action from the last step
    state = self.last_state
    action = self.last_action

    # There is no next state because the game has ended
    next_state = None

    # Get the reward from the events
    reward = reward_from_events(self, events)

    # Store the last transition
    self.transitions.append(Transition(state, action, next_state, reward))

    # Update Q-table for the last transition
    update_q_table(self, state, action, next_state, reward, terminal=True)

    # Save the Q-table (model)
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

def update_q_table(self, state, action, next_state, reward, terminal=False):
    """
    Update the Q-table using the Q-learning update rule.
    """
    # Ensure states are in the Q-table
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    if next_state is not None and next_state not in self.q_table:
        self.q_table[next_state] = np.zeros(len(ACTIONS))

    # Get the index of the action taken
    action_index = ACTIONS.index(action)

    # Q-learning update rule
    if terminal or next_state is None:
        target = reward
    else:
        next_max = np.max(self.q_table[next_state])
        target = reward + self.gamma * next_max

    # Update the Q-value for the state-action pair
    old_value = self.q_table[state][action_index]
    self.q_table[state][action_index] = (1 - self.alpha) * old_value + self.alpha * target

def reward_from_events(self, events: List[str]) -> float:
    """
    Compute the reward for the given events.

    You can modify the rewards to shape the agent's behavior.
    """
    # Default rewards for events
    game_rewards = {
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -3,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: 2,
        e.CRATE_DESTROYED: 5,
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -50,
        e.SURVIVED_ROUND: 20,
        # Add any custom events you have defined
    }

    reward_sum = 0
    for event in events:
        reward_sum += game_rewards.get(event, 0)  # Default to 0 if the event is not in the dict

    return reward_sum
