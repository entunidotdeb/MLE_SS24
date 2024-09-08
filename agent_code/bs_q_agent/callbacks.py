import os
import pickle
import random
import numpy as np
from collections import defaultdict
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # Initialize the Q-table and learning parameters
        self.q_table = dict()
        self.alpha = 0.2
        self.gamma = 0.7
        self.epsilon = 0.1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.last_state = None
        self.last_action = None
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Choose an action based on the Q-learning policy (epsilon-greedy).
    """

    # Get the current state (simplified representation with engineered features)
    state = get_state(self, game_state)
    
    # Epsilon-greedy action selection (explore with probability epsilon)
    if np.random.rand() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(ACTIONS)
    else:
        # Check if the state exists in the Q-table, if not, initialize it with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        
        # Exploit: Use A* pathfinding to suggest the next move toward the nearest coin
        next_step = find_path_to_nearest_coin(self, game_state['field'], game_state['self'][3], game_state['coins'], game_state['bombs'])
        if next_step:
            # Translate the next step into a valid action (UP, DOWN, LEFT, RIGHT)
            next_step = tuple(next_step)
            agent_x, agent_y = game_state['self'][3]
            if next_step == (agent_x, agent_y - 1):
                action = 'UP'
            elif next_step == (agent_x, agent_y + 1):
                action = 'DOWN'
            elif next_step == (agent_x - 1, agent_y):
                action = 'LEFT'
            elif next_step == (agent_x + 1, agent_y):
                action = 'RIGHT'
            else:
                # Fallback to Q-learning action if no valid path is found
                action = ACTIONS[np.argmax(self.q_table[state])]
        else:
            # If no path is found, fallback to the best Q-learning action
            action = ACTIONS[np.argmax(self.q_table[state])]
    
    # Save the last state and action for Q-learning updates
    self.last_state = state
    self.last_action = action
    
    print("ACTION {}".format(action))
    return action

def get_state(self, game_state):
    """
    Converts the game state to a feature vector with engineered features.
    Features:
    1. Agent's (x, y) position
    2. Shortest path to the nearest coin using A* (if available)
    3. Whether a coin is nearby (within a radius of 2)
    """
    
    # Agent's position

    agent_x, agent_y = game_state['self'][3]
    
    # Coin positions as a list of tuples
    coins = [(int(coin[0]), int(coin[1])) for coin in game_state['coins']]
    
    # Feature 2: Find the path to the nearest coin using A*
    if coins:
        path = find_path_to_nearest_coin(self, game_state['field'], (agent_x, agent_y), coins, game_state['bombs'])
        if path:
            direction_feature = get_direction(agent_x, agent_y, path)
        else:
            direction_feature = -1  # No path found
    else:
        direction_feature = -1  # No coins available
    
    # Feature 3: Is a coin nearby (binary feature: 1 if coin within 2 tiles, else 0)
    coin_nearby_feature = is_coin_nearby((agent_x, agent_y), coins, radius=2)

    # Combine all features into a tuple
    feature_vector = (int(agent_x), int(agent_y), direction_feature, coin_nearby_feature)
    
    return feature_vector



def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.
    This is not necessary for Task 1 but can be expanded in later tasks.
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    # Append features like walls, coins, bombs, etc. here if needed in future tasks
    channels.append(...)

    # concatenate them as a feature tensor (they must have the same shape)
    stacked_channels = np.stack(channels)

    # and return them as a vector
    return stacked_channels.reshape(-1)

def save_model(self):
    """Saves the trained model."""
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

def is_coin_nearby(agent_pos, coins, radius=2):
    """Check if there is a coin within a certain radius of the agent."""
    for coin in coins:
        if abs(agent_pos[0] - coin[0]) + abs(agent_pos[1] - coin[1]) <= radius:
            return 1  # Coin nearby
    return 0  # No coin nearby

def convert_arena_to_astar_grid(arena, bombs):
    """
    """
    grid = [[-1 for cols in rows] for rows in arena]

    # consider crates as walls (or a blocked path) too (as of now)
    for i in range(0, len(arena)):
        for j in range(0, len(arena[i])):
            if arena[i][j] == 1:
                grid[i][j] = -2
            elif arena[i][j] == 0:
                grid[i][j] = 1
            else:
                grid[i][j] = int(arena[i][j])

    for (bomb_pos, _) in bombs:
        grid[bomb_pos[0]][bomb_pos[0]] = -3

    return grid


def find_path_to_nearest_coin(self, field, agent_pos, coins, bombs):
    """
    Use A* to find the shortest path to the nearest coin.
    :param game_state: The current game state.
    :return: A list of coordinates representing the path to the nearest coin, or None if no path is found.
    """
    # Agent's position
    agent_x, agent_y = agent_pos
    
    if not coins:
        return None

    # Convert the arena to an A* compatible grid (0 = free, 1 = obstacle)
    grid = convert_arena_to_astar_grid(field, bombs)
    astar_grid_obj = Grid(matrix=grid)
    
    # Find the nearest coin using Manhattan distance as a heuristic
    closest_coins = sorted(coins, key=lambda coin: abs(agent_pos[0] - coin[0]) + abs(agent_pos[1] - coin[1]))
    
    # Use A* to find the shortest path to the nearest coin
    finder = AStarFinder()
    start = astar_grid_obj.node(int(agent_x), int(agent_y))
    end = astar_grid_obj.node(closest_coins[0][0], closest_coins[0][1])
    path, _ = finder.find_path(start, end, astar_grid_obj)
    
    # If a path was found, return the first step toward the coin
    if path:
        if len(path) > 1:
            return path[1]  # Return the first step in the path
        elif len(path) == 1:
            return path[0] # Return the current pos of agent, it means the agent is already on the coin
    return None

def get_direction(agent_x, agent_y, next_step):
    """
    Convert the next step in the path to a directional feature (UP, DOWN, LEFT, RIGHT).
    """
    next_x, next_y = next_step
    if next_x == agent_x and next_y == agent_y:
        print("NEXT X {} NEXT Y {}".format(next_x, next_y))
        return 0 # WAIT 
    elif next_x < agent_x:
        return 4  # LEFT
    elif next_x > agent_x:
        return 2  # RIGHT
    elif next_y < agent_y:
        return 1  # UP
    elif next_y > agent_y:
        return 3  # DOWN
    return -1  # Invalid