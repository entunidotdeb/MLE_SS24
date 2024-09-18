import os
import pickle
import random
import numpy as np
from collections import defaultdict
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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

    # 0 - COIN
    # 1 - AGENT_IN_DANGER
    # 2 - AGENT_IN_DANGER_PREFERRED_COIN
    # 3 - SAFE - NO COIN - NO DANGER
    # 4 - DROP_BOMB
    # 5 - NAVIGATE_TO_CRATE
    INFO = None

    path = None
    direction_feature = None
    state = None

    agent_x, agent_y = game_state['self'][3]

    state, path = get_state(self, game_state)
    
    # Check if the state exists in the Q-table, if not, initialize it with zeros
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    
    if state and path:
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
    elif state and not path:
        action = 'BOMB'
    else:
        # If no path is found, fallback to the best Q-learning action
        action = ACTIONS[np.argmax(self.q_table[state])]
    
    # Save the last state and action for Q-learning updates
    self.last_state = state
    self.last_action = action
    
    print("ACTION {}".format(action))
    return action



def get_state(self, game_state):
    
    agent_x, agent_y = game_state['self'][3]
    closest_coin, dist_to_closest_coint, direction_feature = None, None, None
    state, path = None, None

    field = game_state['field'].tolist()

    new_transformed_grid = convert_arena_to_astar_grid(field, game_state['bombs'], game_state['explosion_map'])

    if new_transformed_grid[agent_x][agent_y] in [-4, -3]:
        state, path = get_danger_state_feature(self, (agent_x, agent_y), new_transformed_grid, game_state['coins'], game_state['bombs'], field)
    elif game_state['coins']:
        closest_coin, path, dist_to_closest_coint = get_closest_coin_from(self, agent_x, agent_y, game_state['coins'], new_transformed_grid)
        if path:
            if len(path) == 1:
                path = path[0]
            else:
                path = path[1]
            direction_feature = get_direction(agent_x, agent_y, path)
            if closest_coin and dist_to_closest_coint and direction_feature:
                state = (agent_x, agent_y, 0, closest_coin[0], closest_coin[1], direction_feature, dist_to_closest_coint)
        elif game_state['self'][2] and crate_exists(field):
            state, path = get_state_for_bomb_drop(self, (agent_x, agent_y), new_transformed_grid, field, game_state['bombs'], game_state['explosion_map'])
    elif game_state['self'][2] and crate_exists(field):
        state, path = get_state_for_bomb_drop(self, (agent_x, agent_y), new_transformed_grid, field, game_state['bombs'], game_state['explosion_map'])
    else:
        path = (agent_x, agent_y)
        direction_feature = get_direction(agent_x, agent_y, path)
        state = (agent_x, agent_y, 3, agent_x, agent_y, direction_feature, dist_to_closest_coint)
    return state, path

def get_safe_node_options(start_pos, grid):
    options = []
    x, y = start_pos
    for val in [-1, 1]:
        if 0 < x + val < len(grid) and grid[x+val][y] > 0:
            options.append((x+val, y))
        if 0 < y + val < len(grid) and grid[x][y + val] > 0:
            options.append((x, y+val))
    return options


def get_safe_all_node_options(self, agent_pos, orig_grid, transformed_grid, bomb):
    all_safe_options = []
    all_safe_options = get_safe_node_options(agent_pos, transformed_grid)

    if all_safe_options:
        return all_safe_options
    
    agent_x, agent_y = agent_pos
    pts = [
        (agent_x + 1, agent_y), 
        (agent_x, agent_y + 1), 
        (agent_x - 1, agent_y), 
        (agent_x, agent_y - 1)
    ]
    filtered_pts = []

    for pt in pts:
        pt_x, pt_y = pt
        if transformed_grid[pt_x][pt_y] == -4 and orig_grid[pt_x][pt_y] == 0:
            filtered_pts.append(pt)

    all_safe_options = sorted(filtered_pts, key=lambda x: len(find_path_to_nearest_coin(self, orig_grid, x, bomb)), reverse=True)
    
    return all_safe_options


def convert_original(field):
    new_field = [[-1 for col in rows] for rows in field]
    for i in range(len(field)):
        for j in range(len(field[i])):
            if field[i][j] == 0:
                new_field[i][j] = 1
            elif field[i][j] == 1:
                new_field[i][j] = -2
            else:
                new_field[i][j] = field[i][j]
    return new_field

def get_danger_state_feature(self, agent_pos, new_transformed_grid, coins, bombs, orig_field):
    state_feature = None
    path = None
    safe_options = []
    agent_x, agent_y = agent_pos
    
    best_safe_option_xy = None
    best_safe_option_closest_coin = None
    best_safe_option_closest_coin_distance = None

    converted_orig_grid = convert_original(orig_field)
    danger_bomb, timer = find_danger_causing_bomb(self, (agent_x, agent_y), bombs, converted_orig_grid)
    safe_options = get_safe_all_node_options(self, agent_pos, orig_field, new_transformed_grid, danger_bomb)

    if not safe_options:
        direction = get_direction(agent_x, agent_y, (agent_x, agent_y))
        state = (agent_x, agent_y, 1, danger_bomb[0], danger_bomb[1], direction, timer)
        path = (agent_x, agent_y)
        return state, path
    
    best_safe_option_xy = safe_options[0]

    if coins:
        best_safe_option_closest_coin_distance = len(new_transformed_grid) * len(new_transformed_grid)
        for option in safe_options:
            res = get_closest_coin_from(self, option[0], option[1], coins, new_transformed_grid)
            option_closest_coin, _, option_closest_coin_distance = res
            if option_closest_coin_distance < best_safe_option_closest_coin_distance:
                best_safe_option_xy = option
                best_safe_option_closest_coin = option_closest_coin
                best_safe_option_closest_coin_distance = option_closest_coin_distance

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

def convert_arena_to_astar_grid(arena, bombs, explosion_map):
    """
    """
    grid = [[-1 for cols in rows] for rows in arena]

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
                elif 0 < new_x < (len(arena) - 1) and grid[new_x][new_y] in [1, -2]:
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
                elif 0 < new_y < (len(arena) - 1) and grid[new_x][new_y] in [1, -2]:
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
                elif 0 < new_x < (len(arena) - 1) and grid[new_x][new_y] in [1, -2]:
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
                elif 0 < new_y < (len(arena) - 1) and grid[new_x][new_y] in [1, -2]:
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

    transposed_matrix = np.array(field).T

    # Convert the arena to an A* compatible grid (0 = free, 1 = obstacle)
    astar_grid_obj = Grid(matrix=transposed_matrix)
    
    # Use A* to find the shortest path to the nearest coin
    finder = AStarFinder()
    start = astar_grid_obj.node(int(agent_x), int(agent_y))
    end = astar_grid_obj.node(int(closest_coin[0]), int(closest_coin[1]))
    path, _ = finder.find_path(start, end, astar_grid_obj)
    
    return path

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

def get_closest_coin_from(self, x, y, coins, grid):
    """ 
    Finds closest coin and its distance from given x,y
    """ 
    closest_coin, closest_path, closest_distance = None, None, len(grid) * len(grid)
    
    if not coins:
        return None, None, None

    for coin in coins:
        if grid[coin[0]][coin[1]] not in [-4, -5]:
            path_to_coin = find_path_to_nearest_coin(self, grid, (x, y), coin)
            if path_to_coin and len(path_to_coin) < closest_distance:
                closest_distance = len(path_to_coin)
                closest_coin = coin
                closest_path = path_to_coin

    return closest_coin, closest_path, closest_distance

def find_danger_causing_bomb(self, agent_pos, bombs, grid):
    bomb_closest, timer = None, None
    max_distance = len(grid) * len(grid)
    
    for bomb in bombs:
        bomb_pos, t = bomb
        path = find_path_to_nearest_coin(self, grid, agent_pos, bomb_pos)
        if path and len(path) < max_distance:
            max_distance = len(path)
            bomb_closest = bomb_pos
            timer = t
    
    return (bomb_closest, timer)

def crate_exists(field):
    exists = False
    for row in field:
        for col in row:
            if col == 1:
                exists = True
                return exists
    return exists


def get_crates_from_grid(grid):
    crates = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == -2:
                crates.append((i, j))
    return crates
    

def get_total_impact_count_crates(grid, crate_pos):
    total_count = 0

    radius = 3
    inc = 1 

    while(inc <= radius):
        if 0 <= crate_pos[0]+inc < len(grid) and 0 <= crate_pos[1] < len(grid):
            if grid[crate_pos[0]+inc][crate_pos[1]] == -1:
                break
            elif grid[crate_pos[0]+inc][crate_pos[1]] == -2:
                total_count+=1
            else:
                pass
            inc+=1
        else:
            break
    
    radius = 3
    inc = 1 

    while(inc <= radius):
        if 0 <= crate_pos[0] < len(grid) and 0 <= crate_pos[1]+inc < len(grid):
            if grid[crate_pos[0]][crate_pos[1]+inc] == -1:
                break
            elif grid[crate_pos[0]][crate_pos[1]+inc] == -2:
                total_count+=1
            else:
                pass
            inc+=1
        else:
            break
    
    radius = -3
    inc = -1

    while(inc >= radius):
        if 0 <= crate_pos[0] < len(grid) and 0 <= crate_pos[1]+inc < len(grid):
            if grid[crate_pos[0]][crate_pos[1]+inc] == -1:
                break
            elif grid[crate_pos[0]][crate_pos[1]+inc] == -2:
                total_count+=1
            else:
                pass
            inc-=1
        else:
            break
    
    radius = -3
    inc = -1

    while(inc >= radius):
        if 0 <= crate_pos[0]+inc < len(grid) and 0 <= crate_pos[1] < len(grid):
            if grid[crate_pos[0]+inc][crate_pos[1]] == -1:
                break
            elif grid[crate_pos[0]+inc][crate_pos[1]] == -2:
                total_count+=1
            else:
                pass
            inc-=1
        else:
            break
    
    return total_count


def get_state_for_bomb_drop(self, agent_pos, grid, orig_grid, bombs, explosion_map):
    distance_to_crate = len(grid) * len(grid)
    crate_to_go_pos = (None, None)
    path_to_crate = []
    max_crates_in_line = -1
    state = None
    final_path = None
    crate_aimed = (None, None)

    all_crates = get_crates_from_grid(grid)

    if not all_crates:
        state = None
        final_path = None
    
    for crate in all_crates:
        for crate_pos in [(crate[0]+1, crate[1]), (crate[0]-1, crate[1]), (crate[0], crate[1]+1), (crate[0], crate[1]-1)]:
            crate_x, crate_y = crate_pos
            if 0 <= crate_x < len(grid) and 0 <= crate_y < len(grid) and grid[crate_x][crate_y] == 1:
                path = []
                test_impact = get_total_impact_count_crates(grid, (crate_x, crate_y))
                if test_impact > max_crates_in_line:
                    path = find_path_to_nearest_coin(self, grid, agent_pos, (crate_x, crate_y))
                if path and len(path) < distance_to_crate:
                    distance_to_crate = len(path)
                    crate_to_go_pos = (crate_x, crate_y)
                    max_crates_in_line = test_impact
                    path_to_crate = path
                    crate_aimed = crate
    
    if distance_to_crate == len(grid) * len(grid):
        state = None
        final_path = None
    elif distance_to_crate == 1 and crate_to_go_pos and path_to_crate:
        direction = 5
        state = (agent_pos[0], agent_pos[1], 4, crate_aimed[0], crate_aimed[1], direction, max_crates_in_line)
        final_path = None
    elif 1 < distance_to_crate < len(grid) * len(grid) and crate_to_go_pos and path_to_crate:
        direction = get_direction(agent_pos[0], agent_pos[1], path_to_crate[1])
        state = (agent_pos[0], agent_pos[1], 5, crate_aimed[0], crate_aimed[1], direction, max_crates_in_line)
        final_path = path_to_crate[1]
    else:
        state = None
        final_path = None
    
    return (state, final_path)


