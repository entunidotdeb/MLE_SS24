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
    self.all_actions = []
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # We don't initialize training parameters here anymore
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Choose an action based on the Q-learning policy (epsilon-greedy).
    """

    # Get state and suggested action from your logic
    state, suggested_action = get_state(self, game_state)
    
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    
    # Epsilon-greedy policy for action selection
    if self.train and random.uniform(0, 1) < self.epsilon:
        # Exploration: choose a random action
        q_action = np.random.choice(ACTIONS)
    else:
        # Exploitation: choose the best action from Q-table
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        # Handle multiple actions with the same max Q-value
        max_actions = [ACTIONS[i] for i, q in enumerate(q_values) if q == max_q]
        q_action = random.choice(max_actions)
    
    if q_action != suggested_action:
        self.logger.debug("q table suggested action {} IMP".format(q_action))
        action = suggested_action
    else:
        action = q_action

    # Decide whether to override Q-learning action with suggested action
    # if state and should_override_q_learning(self, state, q_action, suggested_action, game_state):
    #     action = suggested_action
    # else:
    #     action = q_action
    
    # Save for Q-learning updates
    self.last_state = state
    self.last_action = action
    self.all_actions.append(action)
    
    return action

def should_override_q_learning(self, state, q_action, suggested_action, game_state):
    # Example condition: Always trust your logic when in immediate danger
    if state[2] == 1 or state[2] == 2:
        return True
    # Example condition: If your logic strongly suggests an action
    if suggested_action == 'BOMB' and game_state['self'][2]:
        return True
    
    # Out of Bounds check
    action_x, action_y = get_action_coordinates(game_state['self'][3], q_action)
    if 0 >= action_x or action_x > len(game_state['field']) - 1:
        return True
    
    if 0 >= action_y or action_y > len(game_state['field']) - 1:
        return True
    
    if int(game_state['field'][action_x][action_y]) != 0:
        return True

    if int(game_state['explosion_map'][action_x][action_y]) > 0:
        return True

    for bomb in game_state['bombs']:
        if game_state['self'][3] == bomb[0]:
            return True    

    # Default: Do not override
    return False

def get_adjacent_states(self, agent_pos, grid, coins):
    up_val, left_val, down_val, right_val = None, None, None, None
    agent_x, agent_y = agent_pos
    up = (agent_x, agent_y - 1)
    down = (agent_x, agent_y + 1)
    left = (agent_x - 1, agent_y)
    right = (agent_x + 1, agent_y)

    up_val = get_adjacent_node_values(self, up, grid, coins)
    right_val = get_adjacent_node_values(self, right, grid, coins)
    down_val = get_adjacent_node_values(self, down, grid, coins)
    left_val = get_adjacent_node_values(self, left, grid, coins)

    return (up_val, right_val, down_val, left_val)

def get_adjacent_node_values(self, pos, grid, coins):
    val = -1
    x, y = pos
    read_val = grid[x][y]

    if read_val == 1:
        val = 0
        if (x, y) in coins:
            val = 8
    elif read_val == -1:
        val = 1
    elif read_val == -2:
        val = 2
    elif read_val == -3:
        val = 3
    elif read_val == -4:
        val = 4
        if (x, y) in coins:
            val = 5
    elif read_val == -5:
        val = 6
        if (x, y) in coins:
            val = 7
    else:
        val = -1
    
    return val


def get_state(self, game_state):
    
    agent_x, agent_y = game_state['self'][3]
    closest_coin, dist_to_closest_coint, direction_feature = None, None, None
    state, path = None, None
    action = 'WAIT'

    field = game_state['field'].tolist()

    new_transformed_grid = convert_arena_to_astar_grid(field, game_state['bombs'], game_state['explosion_map'])

    if new_transformed_grid[agent_x][agent_y] in [-4, -3]:
        state, path = get_danger_state_feature(self, (agent_x, agent_y), new_transformed_grid, game_state['coins'], game_state['bombs'], field)
    elif game_state['coins']:
        closest_coin, path, dist_to_closest_coint = get_closest_coin_from(self, agent_x, agent_y, game_state['coins'], new_transformed_grid)
        if path and closest_coin:
            if len(path) == 1:
                path = path[0]
            else:
                path = path[1]
            direction_feature = get_direction(agent_x, agent_y, path)
            up, right, down, left = get_adjacent_states(self, (agent_x, agent_y), new_transformed_grid, game_state['coins'])
            state = (0, direction_feature, dist_to_closest_coint, up, right, down, left)
        elif game_state['self'][2] and crate_exists(field):
            state, path = get_state_for_bomb_drop(self, (agent_x, agent_y), new_transformed_grid, game_state['coins'])
    elif game_state['self'][2] and crate_exists(field):
        state, path = get_state_for_bomb_drop(self, (agent_x, agent_y), new_transformed_grid, game_state['coins'])
    else:
        up, right, down, left = get_adjacent_states(self, (agent_x, agent_y), new_transformed_grid, game_state['coins'])
        path = (agent_x, agent_y)
        state = (4, 0, 0, up, right, down, left)
    if state and path:
        next_step = tuple(path)
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
            action = 'WAIT'
    elif state and not path:
        action = 'BOMB'
    else:
        action = 'WAIT'

    return state, action

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
        if transformed_grid[pt_x][pt_y] == -4 and orig_grid[pt_x][pt_y] == 1:
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

def get_opposite_action(action):
    opp_action = 'INVALID'
    if action == 'UP':
        opp_action = 'DOWN'
    elif action == 'DOWN':
        opp_action = 'UP'
    elif action == 'LEFT':
        opp_action = 'RIGHT'
    elif action == 'RIGHT':
        opp_action = 'LEFT'
    else:
        opp_action = 'INVALID'
    return opp_action

def get_action_coordinates(agent_pos, action):
    x, y = agent_pos
    if action == 'UP':
        y-=1
    elif action == 'DOWN':
        y+=1
    elif action == 'RIGHT':
        x+=1
    elif action == 'LEFT':
        x-=1
    else:
        pass
    return (x, y)

def get_action_before_last_bomb_dropped(self, all_actions):
    action = None
    for i in range(len(all_actions)-1, 0, -1):
        if all_actions[i] == "BOMB":
            action = all_actions[i-1]
            break
    if action == "WAIT" or action == "INVALID" and 0 <= (i-2) < len(all_actions):
        for j in range(i-2, 0, -1):
            if all_actions[j] in ["UP", "LEFT", "RIGHT", "DOWN"]:
                action = all_actions[j]
                break
    return action if action else "INVALID"

def get_danger_state_feature(self, agent_pos, new_transformed_grid, coins, bombs, orig_field):
    state_feature = None
    path = None
    safe_options = []
    agent_x, agent_y = agent_pos
    up, right, down, left = get_adjacent_states(self, agent_pos, new_transformed_grid, coins)
    
    best_safe_option_xy = None
    best_safe_option_closest_coin = None
    best_safe_option_closest_coin_distance = None

    converted_orig_grid = convert_original(orig_field)
    danger_bomb, distance_to_bomb = find_danger_causing_bomb(self, (agent_x, agent_y), bombs, converted_orig_grid)
    if new_transformed_grid[agent_x][agent_y] == -3 and len(self.all_actions) > 1:
        last_action = get_action_before_last_bomb_dropped(self, self.all_actions)
        action = get_opposite_action(self.all_actions[-2])
        new_pos = get_action_coordinates(agent_pos, action)
        safe_options.append(new_pos)
    else:
        safe_options = get_safe_all_node_options(self, agent_pos, converted_orig_grid, new_transformed_grid, danger_bomb)

    if not safe_options:
        if danger_bomb and distance_to_bomb:
            direction = get_direction(agent_x, agent_y, danger_bomb)
            state_feature = (1, direction, distance_to_bomb, up, right, down, left)
        else:
            state_feature = (1, None, None, up, right, down, left)
        path = (agent_x, agent_y)
        return state_feature, path
    
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
        direction = get_direction(agent_x, agent_y, best_safe_option_closest_coin)
        # coin state feature
        state_feature  = (3, direction, distance_to_coin, up, right, down, left)
        path = best_safe_option_xy
    else:
        # bomb state feature
        if danger_bomb:
            direction = get_direction(agent_x, agent_y, danger_bomb)
            state_feature = (1, direction, distance_to_bomb, up, right, down, left)
        else:
            state_feature = (1, None, None, up, right, down, left)
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
    path = []
    
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

    return (closest_coin, closest_path, closest_distance)

def find_danger_causing_bomb(self, agent_pos, bombs, grid):
    bomb_closest = None
    max_distance = len(grid) * len(grid)
    
    for bomb in bombs:
        bomb_pos, _ = bomb
        path = find_path_to_nearest_coin(self, grid, agent_pos, bomb_pos)
        if path and len(path) < max_distance:
            max_distance = len(path)
            bomb_closest = bomb_pos
    
    return (bomb_closest, max_distance)

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


def get_state_for_bomb_drop(self, agent_pos, grid, coins):
    distance_to_crate = len(grid) * len(grid)
    crate_to_go_pos = (None, None)
    path_to_crate = []
    max_crates_in_line = -1
    state = None
    final_path = None
    crate_aimed = (None, None)
    up, right, down, left = get_adjacent_states(self, agent_pos, grid, coins)

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
        direction = get_direction(agent_pos[0], agent_pos[1], crate_aimed)
        state = (5, direction, distance_to_crate, up, right, down, left)
        final_path = None
    elif 1 < distance_to_crate < len(grid) * len(grid) and crate_to_go_pos and path_to_crate:
        direction = get_direction(agent_pos[0], agent_pos[1], path_to_crate[1])
        state = (2, direction, distance_to_crate, up, right, down, left)
        final_path = path_to_crate[1]
    else:
        state = None
        final_path = None
    
    return (state, final_path)


