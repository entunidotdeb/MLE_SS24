from collections import namedtuple, deque
import pickle
from typing import List
import events as e
from .callbacks import get_state, ACTIONS
import numpy as np

# This is only an example!
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- Do modify
TRANSITION_HISTORY_SIZE = 3  # Keep only the last few transitions

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def setup_training(self):
    """
    Initialize self for training purposes.
    This is called after `setup` in callbacks.py.
    """
    # Store transition tuples (state, action, reward, next_state) for Q-learning updates
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    This is where you update your Q-table.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Get the current and next states based on the new `get_state` feature engineering
    old_state = get_state(self, old_game_state) if old_game_state else None
    new_state = get_state(self, new_game_state) if new_game_state else None
    
    # Check for custom events like revisiting positions and exploring new cells
    # Commenting out the custom events logic as requested
    # if new_game_state:
    #     x, y = new_game_state['self'][3]
    #     if (x, y) in self.visited_positions:
    #         events.append('REVISITED_POSITION')  # Custom penalty for revisiting
    #     else:
    #         events.append('NEW_CELL_VISITED')  # Reward for new cell exploration
    #     self.visited_positions.append((x, y))
    #     if len(self.visited_positions) > 20:  # Keep track of recent positions
    #         self.visited_positions.pop(0)

    # Calculate the reward based on the current game events
    reward = reward_from_events(self, events)

    # Q-learning update: if old_state exists, update Q-table
    if old_state is not None:
        if old_state not in self.q_table:
            self.q_table[old_state] = np.zeros(len(ACTIONS))
        if new_state is not None and new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(len(ACTIONS))

        # Get the index of the action taken
        action_index = ACTIONS.index(self_action)
        
        # Bellman update: Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
        best_future_q = np.max(self.q_table[new_state]) if new_state is not None else 0
        self.q_table[old_state][action_index] = self.q_table[old_state][action_index] + self.alpha * (
            reward + self.gamma * best_future_q - self.q_table[old_state][action_index]
        )
    
    # Store the transition for future reference
    self.transitions.append(Transition(old_state, self_action, new_state, reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent dies to hand out final rewards.
    This replaces game_events_occurred in this round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Get the final state and reward
    final_state = get_state(self, last_game_state) if last_game_state else None
    reward = reward_from_events(self, events)

    # Q-learning update for the last action
    if final_state is not None:
        if final_state not in self.q_table:
            self.q_table[final_state] = np.zeros(len(ACTIONS))
        
        action_index = ACTIONS.index(last_action)
        self.q_table[final_state][action_index] += self.alpha * (reward - self.q_table[final_state][action_index])

    # Save the model (Q-table) at the end of the round
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards to encourage/discourage certain behaviors.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,  # High reward for collecting coins
        e.WAITED: -0.1,  # Penalty for waiting
        e.INVALID_ACTION: -1,  # Penalty for invalid actions
        # Commented out custom events for now
        # 'REVISITED_POSITION': -0.5,  # Custom penalty for revisiting the same position
        # 'NEW_CELL_VISITED': 0.2  # Reward for exploring new cells
    }

    # Sum up the rewards based on the events that occurred
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
