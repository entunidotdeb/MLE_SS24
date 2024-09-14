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


def get_safe_node_options(start_pos, grid):
    options = []
    x, y = start_pos
    for val in [-1, 1]:
        if 0 < x + val < len(grid) - 1 and grid[x+val][y] > 0:
            options.append((x+val, y))
        if 0 < y + val < len(grid) - 1 and grid[x][y + val] > 0:
            options.append((x, y+val))
    return options
    


def act(self, game_state: dict) -> str:
    """
    Choose an action based on the Q-learning policy (epsilon-greedy).
    """

    # 0 - COIN
    # 1 - AGENT_IN_DANGER
    # 2 - AGENT_IN_DANGER_PREFERRED_COIN
    # 3 - SAFE - NO COIN - NO DANGER
    INFO = None

    path, closest_coin, dist_to_closest_coint = None, None, None
    direction_feature = None
    state = None
    best_safe_option = None

    agent_x, agent_y = game_state['self'][3]

    new_transformed_grid = convert_arena_to_astar_grid(game_state['field'], game_state['bombs'], game_state['self'][3], game_state['explosion_map'])

    # check if agent is in danger zone
    if new_transformed_grid[agent_x][agent_y] == -4:
        state, path = get_danger_state_feature(self, agent_x, agent_y, new_transformed_grid, game_state['coins'], game_state['bombs'])
    elif game_state['coins']:
        closest_coin_sorted_array = get_closest_coin_from(agent_x, agent_y, game_state['coins'])
        if closest_coin_sorted_array:
            closest_coin, dist_to_closest_coint = closest_coin_sorted_array[0]
        if game_state['coins']:
            path = find_path_to_nearest_coin(self, new_transformed_grid, game_state['self'][3], closest_coin)
        if path:
            direction_feature = get_direction(agent_x, agent_y, path)
        if closest_coin:
            state = (agent_x, agent_y, 0, closest_coin[0], closest_coin[1], direction_feature, dist_to_closest_coint)
    else:
        path = (agent_x, agent_y)
        direction_feature = get_direction(agent_x, agent_y, path)
        state = (agent_x, agent_y, 3, agent_x, agent_y, direction_feature, dist_to_closest_coint)
    
    # Epsilon-greedy action selection (explore with probability epsilon)
    if np.random.rand() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(ACTIONS)
    else:
        # Check if the state exists in the Q-table, if not, initialize it with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        
        if path:
            # Translate the next step into a valid action (UP, DOWN, LEFT, RIGHT)
            next_step = tuple(path)
            agent_x, agent_y = game_state['self'][3]
            if next_step == (agent_x, agent_y):
                action = 'WAIT'
            elif next_step == (agent_x, agent_y - 1):
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

def get_danger_state_feature(self, agent_x, agent_y, new_transformed_grid, coins, bombs):
    state_feature = None
    path = None
    safe_options = []
    
    safe_options_sorted_with_coins = []
    safe_options_with_closest = []
    best_safe_option_xy = None
    best_safe_option_closest_coin = None
    best_safe_option_closest_coin_distance = None

    danger_bomb, timer = find_danger_causing_bomb(agent_x, agent_y, bombs, new_transformed_grid)
    safe_options = get_safe_node_options((agent_x, agent_y), new_transformed_grid)
    if not safe_options and danger_bomb:
        pts = [(agent_x + 1, agent_y), (agent_x, agent_y + 1), (agent_x - 1, agent_y), (agent_x, agent_y - 1)]
        filtered_pts = []
        for pt in pts:
            pt_x, pt_y = pt
            if new_transformed_grid[pt_x][pt_y] in [-4, 1]:
                filtered_pts.append(pt)
        filtered_pts = sorted(filtered_pts, key=lambda x: manhattan_distance(danger_bomb, x), reverse=True)
        if filtered_pts:
            safe_options.append(filtered_pts[0])

    if not safe_options:
        direction = get_direction(agent_x, agent_y, (agent_x, agent_y))
        state = (agent_x, agent_y, 1, danger_bomb[0], danger_bomb[1], direction, timer)
        path = (agent_x, agent_y)
        return state, path
    
    best_safe_option_xy = safe_options[0]

    if coins:
        for option in safe_options:
            option_closest, distance = get_closest_coin_from(option[0], option[1], coins)[0]
            safe_options_with_closest.append( (option, option_closest, distance) )
        safe_options_sorted_with_coins = sorted(safe_options_with_closest, key=lambda option: option[2])
        if safe_options_sorted_with_coins:
            best_safe_option_xy, best_safe_option_closest_coin, best_safe_option_closest_coin_distance = safe_options_sorted_with_coins[0]

    if best_safe_option_closest_coin and best_safe_option_closest_coin_distance and best_safe_option_xy:
        coin_x = best_safe_option_xy[0]
        coin_y = best_safe_option_xy[1]
        distance_to_coin = best_safe_option_closest_coin_distance
        direction = get_direction(agent_x, agent_y, best_safe_option_xy)
        # coin state feature
        state_feature  = (agent_x, agent_y, 2, coin_x, coin_y, direction, distance_to_coin)
        path = best_safe_option_xy
    else:
        direction = get_direction(agent_x, agent_y, best_safe_option_xy)
        # bomb state feature
        state_feature = (agent_x, agent_y, 1, danger_bomb[0], danger_bomb[1], direction, timer)
        path = best_safe_option_xy
    
    return state_feature, path

def save_model(self):
    """Saves the trained model."""
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

def convert_arena_to_astar_grid(arena, bombs, agent, explosion_map):
    """
    """
    grid = [[-1 for cols in rows] for rows in arena]

    # consider crates as walls (or a blocked path) too (as of now)
    for i in range(0, len(arena)):
        for j in range(0, len(arena[i])):
            # crates
            if arena[i][j] == 1:
                grid[i][j] = -2
            # free node
            elif arena[i][j] == 0:
                grid[i][j] = 1
            else:
                # Stone Walls
                grid[i][j] = int(arena[i][j])
            if explosion_map[i][j] > 0:
                grid[i][j] = -5

    for (bomb_pos, t_left) in bombs:
        bomb_x, bomb_y =  bomb_pos
        grid[bomb_x][bomb_y] = -3
        
        radius, counter = 3, 1
        while (counter <= radius):
            new_x = bomb_x + counter
            new_y = bomb_y
            # check out of bound
            if 0 < new_x < (len(arena) - 1):
                # check if the new tile is wall or not, if yes, then break out of loop
                # not required to further check and update
                if grid[new_x][new_y] == -1:
                    break
                elif 0 < new_x < (len(arena) - 1) and grid[new_x][new_y] > 0:
                    grid[new_x][new_y] = -4
            else:
                break
            counter+=1

        counter = 1
        while (counter <= radius):
            new_x = bomb_x
            new_y = bomb_y + counter
            if 0 < new_y < (len(arena) - 1):
                # check if the new tile is wall or not, if yes, then break out of loop
                # not required to further check and update
                if grid[new_x][new_y] == -1:
                    break
                elif 0 < new_y < (len(arena) - 1) and grid[new_x][new_y] > 0:
                    grid[new_x][new_y] = -4
            else:
                break
            counter+=1
        
        counter, radius = -1, -3
        while (radius <= counter):
            new_x = bomb_x + counter
            new_y = bomb_y
            if 0 < new_x < (len(arena) - 1):
                # check if the new tile is wall or not, if yes, then break out of loop
                # not required to further check and update
                if grid[new_x][new_y] == -1:
                    break
                elif 0 < new_x < (len(arena) - 1) and grid[new_x][new_y] > 0:
                    grid[new_x][new_y] = -4
            else:
                break
            counter-=1
        
        counter = -1
        while (radius <= counter):
            new_x = bomb_x
            new_y = bomb_y + counter
            if 0 < new_y < (len(arena) - 1):
                # check if the new tile is wall or not, if yes, then break out of loop
                # not required to further check and update
                if grid[new_x][new_y] == -1:
                    break
                elif 0 < new_y < (len(arena) - 1) and grid[new_x][new_y] > 0:
                    grid[new_x][new_y] = -4
            else:
                break
            counter-=1
    
    # -3 BOMBS
    # -2 CRATES
    # -1 WALLS
    # 1 FREE

    return grid


def find_path_to_nearest_coin(self, field, agent_pos, closest_coin):
    """
    Use A* to find the shortest path to the nearest coin.
    :param game_state: The current game state.
    :return: A list of coordinates representing the path to the nearest coin, or None if no path is found.
    """
    # Agent's position
    agent_x, agent_y = agent_pos
    path = None
    
    if not closest_coin:
        return path

    # Convert the arena to an A* compatible grid (0 = free, 1 = obstacle)
    astar_grid_obj = Grid(matrix=field)
    
    # Use A* to find the shortest path to the nearest coin
    finder = AStarFinder()
    start = astar_grid_obj.node(int(agent_x), int(agent_y))
    end = astar_grid_obj.node(int(closest_coin[0]), int(closest_coin[1]))
    path, _ = finder.find_path(start, end, astar_grid_obj)
    
    # If a path was found, return the first step toward the coin
    if path:
        if len(path) > 1:
            # Return the first step in the path
            return path[1]
        elif len(path) == 1:
            # Return the current pos of agent, it means the agent is already on the coin
            return path[0]
    
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


def manhattan_distance(a, b):
    """
    Finds manhattan distance btw 2 pts
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_closest_coin_from(x, y, coins):
    """ 
    Finds closest coin and its distance from given x,y
    """ 
    sorted_coin_distances = []
    
    if not coins:
        return sorted_coin_distances
    
    coin_distances_array = [(coin, manhattan_distance((x,y), coin)) for coin in coins]
    sorted_coin_distances = sorted(coin_distances_array, key=lambda x: x[1])

    return sorted_coin_distances

def find_danger_causing_bomb(agent_x, agent_y, bombs, grid):
    bomb_closest, timer = None, None
    
    counter, radius = 1, 3
    while(counter <= radius and not bomb_closest):
        if 0 < agent_x + counter < len(grid) - 1:
            if (grid[agent_x + counter][agent_y] == -1):
                break
            elif (grid[agent_x + counter][agent_y] == -3):
                bomb_closest = (agent_x + counter, agent_y)
            else:
                pass
        counter+=1
    
    counter = 1
    while(counter <= radius and not bomb_closest):
        if 0 < agent_y + counter < len(grid) - 1:
            if (grid[agent_x][agent_y + counter] == -1):
                break
            elif (grid[agent_x][agent_y + counter] == -3):
                bomb_closest = (agent_x, agent_y + counter)
            else:
                pass
        counter+=1
    
    counter, radius = -1, -3
    while(radius <= counter and not bomb_closest):
        if 0 < agent_x + counter < len(grid) - 1:
            if (grid[agent_x + counter][agent_y] == -1):
                break
            elif (grid[agent_x + counter][agent_y] == -3):
                bomb_closest = (agent_x + counter, agent_y)
            else:
                pass
        counter-=1
    
    counter, radius = -1, -3
    while(radius <= counter and not bomb_closest):
        if 0 < agent_y + counter < len(grid) - 1:
            if (grid[agent_x][agent_y + counter] == -1):
                break
            elif (grid[agent_x][agent_y + counter] == -3):
                bomb_closest = (agent_x, agent_y + counter)
            else:
                pass
        counter-=1
    
    if bomb_closest:
        for bomb in bombs:
            bomb_pos, t = bomb
            if bomb_pos == bomb_closest:
                timer = t

    return (bomb_closest, timer)


        




    
