from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
import pdb
import copy
import numpy as np
import random as random
import logging

class CustomEnvReset:

    def __init__(self, env_name, all_factors):

        custom_reset = {
            'DoorKey': self._custom_reset_doorkey, 
            'LavaCrossing': self._custom_reset_lavacrossing, 
            'FourRooms': self._custom_reset_fourrooms, 
            'Empty': self._custom_reset_empty,
            'CartPole': self._custom_reset_cartpole
        }
        
        for k in custom_reset.keys():
            if k in env_name:
                self.factored_reset = custom_reset[k]
        
        self.all_factors = set(all_factors)
    
    def check_valid_factors(self, env, controlled_factors, strict_check=True):
        
        #populate empty set with all controlled factors to check
        occupied_locations = set([])

        for k in controlled_factors.keys():
            
            controlled_val = controlled_factors[k]

            if type(controlled_val) == list:

                #check 0: if the location is within bounds
                if controlled_val[0] >= env.unwrapped.width or controlled_val[0] < 0:
                    return False, 'controlled_reset.py: Cannot have location outside grid bounds'
                if controlled_val[1] >= env.unwrapped.height or controlled_val[1] < 0:
                    return False, 'controlled_reset.py: Cannot have location outside grid bounds'

                #check 1: if the location is in the wall border
                if isinstance(env.unwrapped.grid.get(controlled_val[0], controlled_val[1]), Wall):
                    return False, 'controlled_reset.py: Cannot have location overlapping a wall'
                
                #check 2: check if the location is within occupied locations
                if tuple((controlled_val[0], controlled_val[1])) in occupied_locations:
                    return False, 'controlled_reset.py: Cannot have location be occupied'
                
                #other checks: for door add entire column to set & for any other position add only position the set
                if k == 'door_pos':
                    for i in range(env.unwrapped.height):
                        if (controlled_val[0], i) in occupied_locations:
                            
                            return False, 'controlled_reset.py: Cannot have door column overlapping'
                        occupied_locations.add((controlled_val[0],i))
                
                occupied_locations.add((controlled_val[0], controlled_val[1]))

        #check 3: specific to doorkey, if door is open and key is not held
        if (strict_check) and ('door_open' in controlled_factors and controlled_factors['door_open']) and ('door_locked' in controlled_factors and controlled_factors['door_locked']):
            return False, 'controlled_reset.py: Cannot have door open and locked at same time'
        
        #check 4: specific to doorkey, if holding key and key_pos is part of controlled factors
        if (strict_check) and ('holding_key' in controlled_factors and controlled_factors['holding_key']) and ('key_pos' in controlled_factors):
            
            return False, 'controlled_reset.py: Cannot have agent hold a key but also specify key position'
        
        #check 5: specific to doorkey, if door is locked, the key position should be to left or the key should be held
        if (strict_check) and ('door_locked' in controlled_factors and controlled_factors['door_locked']) and ('door_pos' in controlled_factors) and (('agent_pos' in controlled_factors and controlled_factors['agent_pos'][0] < controlled_factors['door_pos'][0] and 'key_pos' in controlled_factors and controlled_factors['key_pos'][0] > controlled_factors['door_pos'][0]) or ('agent_pos' in controlled_factors and controlled_factors['agent_pos'][0] > controlled_factors['door_pos'][0] and 'key_pos' in controlled_factors and controlled_factors['key_pos'][0] < controlled_factors['door_pos'][0])):
            return False, 'controlled_reset.py: Cannot have door be locked yet the key and agent opposite sides of the door'
         
        return True, None

    def _custom_reset_cartpole(self, env, controlled_factors ={}):
        raise NotImplementedError()
    def _custom_reset_doorkey(self, env, width, height, controlled_factors={}, strict_check=True):
        
        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        # Create an empty grid
        env.unwrapped.grid = Grid(width, height)

        # Generate the surrounding walls
        env.unwrapped.grid.wall_rect(0, 0, width, height)

        #check if the controlled factors is valid or not
        valid, error = self.check_valid_factors(env, controlled_factors, strict_check=strict_check)

        if not valid:
            raise Exception(f'ERROR: the factors {controlled_factors} are not valid | {error}')
        
       

        # Used locations
        used_locations = set([(0,0)])
        used_columns = set([0])


        #all set factor values
        all_factors = {}

        #insert controlled factors in to the set of all factors
        for f in controlled_factors.keys():
            
            factor = controlled_factors[f]

            if type(factor) == list:

                #add location to used set
                used_locations.add(tuple(factor))

                #add column for door
                if f == 'door_pos':
                    for i in range(env.unwrapped.height):
                        used_locations.add((factor[0],i))
            

            all_factors[f] = factor
        
        #randomly set the factor values for all other factors: sample until they are not in used locations
        remaining_factors = list(sorted(self.all_factors - set(list(controlled_factors.keys()))))

        for f in remaining_factors:

            if f == 'door_pos':

                rand_door_loc = (0,0)
                
                while (rand_door_loc in used_locations) or (rand_door_loc[0] in used_columns) or (isinstance(env.unwrapped.grid.get(rand_door_loc[0], rand_door_loc[1]), Wall)):
                    
                    rand_door_loc = (env.unwrapped._rand_int(2, width - 2), env.unwrapped._rand_int(1, height - 2))
                    # print("rand_door_loc:", rand_door_loc)
                    
                
                all_factors[f] = rand_door_loc
                
                used_locations.add(rand_door_loc)
                for i in range(env.unwrapped.height):
                    used_locations.add((rand_door_loc[0],i))
                
                used_columns.add(rand_door_loc[0])
                

            if f == 'goal_pos':
                rand_goal_loc = (0,0)
                
                while (rand_goal_loc in used_locations) or (isinstance(env.unwrapped.grid.get(rand_goal_loc[0], rand_goal_loc[1]), Wall)):
                    
                    rand_goal_loc =  (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_goal_loc
                used_locations.add(rand_goal_loc)
                used_columns.add(rand_goal_loc[0])
                
            
            if f == 'door_locked':

                all_factors[f] = True if random.random() <=0.5 else False
            
            if f == 'door_open':
                all_factors[f] = True if random.random() <= 0.5 else False
            
            if f == 'agent_pos':
                rand_agent_loc = (0,0)
                
                while (rand_agent_loc in used_locations) or (isinstance(env.unwrapped.grid.get(rand_agent_loc[0], rand_agent_loc[1]), Wall)):
                    
                    rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_agent_loc
                used_locations.add(rand_agent_loc)
                used_columns.add(rand_agent_loc[0])
                

            if f == 'agent_dir':
                all_factors[f] = env.unwrapped._rand_int(0, 4)

            if f == 'holding_key' and ('holding_key' not in all_factors):
                all_factors[f] = True if random.random()<=0.5 else False
            
            if f == 'key_pos':

                #special case: set the holding_key attribute in case not set before
                if 'holding_key' not in all_factors:
                    all_factors['holding_key'] = True if random.random()<=0.5 else False
                
                #set key location anywhere if not holding and door open
                if not all_factors['holding_key'] and all_factors['door_open']:
                    rand_key_loc = (0,0)
                    
                    while (rand_key_loc  in used_locations) or isinstance(env.unwrapped.grid.get(rand_key_loc[0], rand_key_loc[1]), Wall):
                        
                        rand_key_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                    

                    used_locations.add(rand_key_loc)
                    used_columns.add(rand_key_loc[0])
                #set key location to left half if not holding and door not open
                elif not all_factors['holding_key'] and not all_factors['door_open']:
                    rand_key_loc = (0,0)

                    #align key position so that it is on the same side of the door as the agent is
                    min_col = 1 if all_factors['agent_pos'][0] < all_factors['door_pos'][0] else all_factors['door_pos'][0] + 1
                    max_col = all_factors['door_pos'][0] - 1 if all_factors['agent_pos'][0] < all_factors['door_pos'][0] else width-1
                    
                    while (rand_key_loc  in used_locations) or isinstance(env.unwrapped.grid.get(rand_key_loc[0], rand_key_loc[1]), Wall):
                        
                        rand_key_loc = (env.unwrapped._rand_int(min_col, max_col) if min_col!=max_col else min_col, env.unwrapped._rand_int(1, height - 1))
                    

                    used_locations.add(rand_key_loc)
                    used_columns.add(rand_key_loc[0])
                #set key location to none if holding
                elif all_factors['holding_key']:

                    rand_key_loc = (None, None)
                
                all_factors[f] = rand_key_loc
        
        
        # factor 1: add goal position 
        env.unwrapped.put_obj(Goal(), all_factors['goal_pos'][0], all_factors['goal_pos'][1])
        assert 'goal' in set([x.type if x is not None else None for x in env.unwrapped.grid.grid]), 'No GOAL!'
        
        # factor 2, 3, 4: add door position, with locked/unlocked and open/closed settings
        splitIdx = all_factors['door_pos'][0]; doorIdx = all_factors['door_pos'][1]
        door_locked = all_factors['door_locked']
        door_open = all_factors['door_open']
        env.unwrapped.grid.vert_wall(splitIdx, 0)
        env.unwrapped.put_obj(Door("yellow", is_locked=door_locked, is_open=door_open), splitIdx, doorIdx)
        # factor 5: add key position and holding
        # factor 6: control holding key
        
        if all_factors['key_pos'] != (None, None):
            
            
            env.unwrapped.put_obj(Key("yellow"), all_factors['key_pos'][0], all_factors['key_pos'][1])
            
            assert 'key' in set([x.type if x is not None else None for x in env.unwrapped.grid.grid]), 'No GOAL!'
        
        else:
            env.unwrapped.carrying = Key("yellow")
        
        # factor 7, 8: add agent position and direction
        # agent_top = all_factors['agent_pos']
        # agent_size = (0,0)
        # env.unwrapped.place_agent(top=agent_top, size=agent_size)
        env.unwrapped.agent_pos = all_factors['agent_pos']
        
        env.unwrapped.agent_dir = all_factors['agent_dir']
        env.unwrapped.mission = "use the key to open the door and then get to the goal"
        
        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng
        return env

    def _custom_reset_empty(self, env, width, height, controlled_factors={},  strict_check=True):
        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        # Create an empty grid
        env.unwrapped.grid = Grid(width, height)

        # Generate the surrounding walls
        env.unwrapped.grid.wall_rect(0, 0, width, height)

        #check if the controlled factors is valid or not
        valid, error = self.check_valid_factors(env, controlled_factors, strict_check= strict_check)

        if not valid:
            raise Exception(f'ERROR: the factors {controlled_factors} are not valid | {error}')
        
        # Used locations
        used_locations = set([(0,0)])


        #all set factor values
        all_factors = {}



        #insert controlled factors in to the set of all factors
        for f in controlled_factors.keys():
            
            factor = controlled_factors[f]

            if type(factor) == list:

                #add location to used set
                used_locations.add(tuple(factor))
            

            all_factors[f] = factor
        

        #randomly set the factor values for all other factors: sample until they are not in used locations
        remaining_factors = self.all_factors - set(list(controlled_factors.keys()))

        for f in remaining_factors:

            if f == 'goal_pos':
                rand_goal_loc = (0,0)

                while (rand_goal_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_goal_loc[0], rand_goal_loc[1]), Wall)):
                    rand_goal_loc =  (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_goal_loc
                used_locations.add(rand_goal_loc)
            
            
            if f == 'agent_pos':
                rand_agent_loc = (0,0)

                while (rand_agent_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_agent_loc[0], rand_agent_loc[1]), Wall)):
                    rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_agent_loc
                used_locations.add(rand_agent_loc)

            if f == 'agent_dir':
                all_factors[f] = env.unwrapped._rand_int(0, 4)

        # Place a goal square in the bottom-right corner
        env.unwrapped.put_obj(Goal(), all_factors['goal_pos'][0], all_factors['goal_pos'][1])

        # Place the agent
        env.unwrapped.agent_pos = all_factors['agent_pos']
        env.unwrapped.agent_dir = all_factors['agent_dir']
        

        env.unwrapped.mission = "get to the green goal square"
        return env
        
    def _custom_reset_fourrooms(self, env, width, height, controlled_factors={},  strict_check=True):

        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        # Create the grid
        env.unwrapped.grid = Grid(width, height)

        # Generate the surrounding walls
        env.unwrapped.grid.horz_wall(0, 0)
        env.unwrapped.grid.horz_wall(0, height - 1)
        env.unwrapped.grid.vert_wall(0, 0)
        env.unwrapped.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    env.unwrapped.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, env.unwrapped._rand_int(yT + 1, yB))
                    env.unwrapped.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    env.unwrapped.grid.horz_wall(xL, yB, room_w)
                    pos = (env.unwrapped._rand_int(xL + 1, xR), yB)
                    env.unwrapped.grid.set(*pos, None)
        
        #check if the controlled factors is valid or not
        valid, error = self.check_valid_factors(env, controlled_factors,  strict_check=strict_check)

        if not valid:
            raise Exception(f'ERROR: the factors {controlled_factors} are not valid | {error}')
        

        # Used locations
        used_locations = set([(0,0)])


        #all set factor values
        all_factors = {}



        #insert controlled factors in to the set of all factors
        for f in controlled_factors.keys():
            
            factor = controlled_factors[f]

            if type(factor) == list:

                #add location to used set
                used_locations.add(tuple(factor))
            

            all_factors[f] = factor
        

        #randomly set the factor values for all other factors: sample until they are not in used locations
        remaining_factors = self.all_factors - set(list(controlled_factors.keys()))

        for f in remaining_factors:

            if f == 'goal_pos':
                rand_goal_loc = (0,0)

                while (rand_goal_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_goal_loc[0], rand_goal_loc[1]), Wall)):
                    rand_goal_loc =  (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_goal_loc
                used_locations.add(rand_goal_loc)
            
            
            if f == 'agent_pos':
                rand_agent_loc = (0,0)

                while (rand_agent_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_agent_loc[0], rand_agent_loc[1]), Wall)):
                    rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_agent_loc
                used_locations.add(rand_agent_loc)

            if f == 'agent_dir':
                all_factors[f] = env.unwrapped._rand_int(0, 4)


        

        # factor 1 / 2: set agent position and direction
        agent_top = all_factors['agent_pos']
        agent_size = (0,0)
        env.unwrapped.place_agent(top=agent_top, size=agent_size)
        env.unwrapped.agent_dir = all_factors['agent_dir']
        

        #factor 3: set the goal position
        env.unwrapped.put_obj(Goal(), *all_factors['goal_pos'])
       
        env.unwrapped.mission = "reach the goal"

        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng
        return env


    def _custom_reset_lavacrossing(self, env, width, height, controlled_factors={},  strict_check=True):

        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        env.unwrapped.grid = Grid(width, height)
        env.unwrapped.grid.wall_rect(0, 0, width, height)

        assert width >= 5 and height >= 5

        #check if the controlled factors is valid or not
        valid, error = self.check_valid_factors(env, controlled_factors,  strict_check=strict_check)

        if not valid:
            raise Exception(f'ERROR: the factors {controlled_factors} are not valid | {error}')
        

        

        # Used locations
        used_locations = set([(0,0)])


        #all set factor values
        all_factors = {}

        #insert controlled factors in to the set of all factors
        for f in controlled_factors.keys():
            
            factor = controlled_factors[f]

            if type(factor) == list:

                #add location to used set
                used_locations.add(tuple(factor))
            

            all_factors[f] = factor
        

        #randomly set the factor values for all other factors: sample until they are not in used locations
        remaining_factors = self.all_factors - set(list(controlled_factors.keys()))

        for f in remaining_factors:

            if f == 'goal_pos':
                rand_goal_loc = (0,0)

                while (rand_goal_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_goal_loc[0], rand_goal_loc[1]), Wall)):
                    rand_goal_loc =  (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_goal_loc
                used_locations.add(rand_goal_loc)
            
            if f == 'agent_pos':
                rand_agent_loc = (0,0)

                while (rand_agent_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_agent_loc[0], rand_agent_loc[1]), Wall)):
                    rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_agent_loc
                used_locations.add(rand_agent_loc)

            if f == 'agent_dir':
                all_factors[f] = env.unwrapped._rand_int(0, 4)

           

        

        # factor 1/2: set agent position and orientation
        env.unwrapped.agent_pos =  all_factors['agent_pos']
        env.unwrapped.agent_dir = all_factors['agent_dir'] 


        # factor 3: set goal position
        env.unwrapped.goal_pos = all_factors['goal_pos']
        env.unwrapped.put_obj(Goal(), *env.unwrapped.goal_pos)

        # Generate and store random gap position
        env.unwrapped.gap_pos = np.array(
            (
                env.unwrapped._rand_int(2, width - 2),
                env.unwrapped._rand_int(1, height - 1),
            )
        )

        # Place the obstacle wall
        env.unwrapped.grid.vert_wall(env.unwrapped.gap_pos[0], 1, height - 2, Lava)

        # Put a hole in the wall
        env.unwrapped.grid.set(*env.unwrapped.gap_pos, None)

        env.unwrapped.mission = (
            "avoid the lava and get to the green goal square"
        )

        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng
        
        return env
    