# import gym_minigrid
# from gym_minigrid.wrappers import  FullyObsWrapper
# from gym_minigrid.wrappers import ImgObsWrapper

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Door, Goal, Key, Box, Ball
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ActionBonus, ObservationWrapper
from gymnasium.wrappers import TimeLimit
from collections.abc import Iterable
from omegaconf import DictConfig

import gymnasium as gym
# from gymnasium import Env
# import gym
import random
import os
from PIL import Image
import pdb
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_augmentor import DataAugmentor
from data.utils.controlled_reset import CustomEnvReset
from matplotlib import pyplot as plt
from typing import Optional
# from custom_env.blockeddoorkey import BlockedDoorKeyEnv
from gymnasium.envs.registration import register


'''
[DONE] TODO: implement (multiple) controlled environment factors at once 
    - Random bug sometimes does not allow for factored expert state to be retrieved 
'''

class FramePadder:
    def __init__(self, target_size: Optional[list]=None):
        assert type(target_size) is not str, f"ERROR in FramePadder __init__: target_size cannot be string type {target_size}"
        self.target_size = tuple(target_size) if target_size else target_size

    def pad(self, frame):
        if self.target_size:
            h, w, c = frame.shape
            target_h, target_w = self.target_size
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            frame = np.pad(frame, 
                            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                            mode='constant', constant_values=0)
        return frame

class StochasticActionWrapper(gym.ActionWrapper):
    """
    Add stochasticity to the actions

    If a random action is provided, it is returned with probability `1 - prob`.
    Else, a random action is sampled from the action space.
    """

    def __init__(self, env=None, prob=0.9, random_action=None):
        super().__init__(env)
        self.prob = prob

    def action(self, action):
        """ """
        
        random = np.random.uniform()
        if random >= self.prob:
            return action
        else:
            return self.env.action_space.sample()

class DataGenerator(gym.Env):
    def __init__(self):
        pass
    def load_config(self, file_path):
        pass
    def _create_expert_observation_space(self, state_attribute_types):
        pass
    def _create_observation_space(self,state_attribute_types):
        pass
    def get_low_high_attr(self, attr):
        pass
    def render(self):
        pass
    def step(self, action):
        pass
    def reset(self):
        pass
    def get_curr_obs(self):
        pass
    def factorize_obs(self, observation):
        pass
    def _construct_state(self):
        pass
    def _get_obs(self, image=None, state)

class MiniGridDataGenerator(DataGenerator):

    def __init__(self, cfg: DictConfig):

        super(MiniGridDataGenerator, self).__init__()

        #register BlockedDoorKey to the environment list
        register(
            id="MiniGrid-BlockedDoorKey-6x6",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 6},
        )
        register(
            id="MiniGrid-BlockedDoorKey-8x8",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 8},
        )
        register(
            id="MiniGrid-BlockedDoorKey-16x16",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 16},
        )

        # store configs from terminal input
        configs = cfg
        
        # Configs from the yaml file
        self.observation_type = configs['observation_space']
        self.state_attributes = configs['state_attributes']
        self.state_attribute_types = configs['state_attribute_types']
        self.reset_type = configs['reset_type']
        self.normalize_state = configs['normalize_state']

        # Create the environment
        self.env = gym.make(configs['environment_name'], render_mode='rgb_array', max_episode_steps = configs['max_steps'], max_steps=configs['max_steps'])

        # Remove the white background from the visual environment
        self.highlight = configs['highlight']

        # Optionally wrap the environment for fully observable states
        self.env = FullyObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)

        
        # Wrap the environment to enable stochastic actions
        if configs['deterministic_action'] is False:
            self.env = StochasticActionWrapper(env=self.env, prob=configs['action_stochasticity'])
        
        # Wrap the environment to enable exploration bonus if needed
        if configs['exploration_bonus'] is True:
            self.env = ActionBonus(env = self.env)

        #Store the controlled factors array
        self.controlled_factors = configs['controlled_factors']
        
        self.render_mode = 'rgb_array'

        #storing the custom reset function if needed
        self.custom_resetter = CustomEnvReset(configs['environment_name'], configs['state_attributes'])

        # Reset the environment to initialize it
        self.env.reset()
        
        #prepare the data augmentor and padding function instance 
        self.data_augmentor = DataAugmentor(configs['transformations'])
        self.padding_func = FramePadder(configs['padded_size'])

        #creating the observation space and actions space
        self.action_space = self.env.action_space
        self.observation_space, self.expert_observation_space = self._create_observation_space(configs['state_attribute_types'])


        #creating other gym environment attributes
        self.spec = self.env.spec
        self.metadata = self.env.metadata
        self.np_random = self.env.np_random
        
    def _create_expert_observation_space(self, state_attribute_types):

        self.gym_space_params = {'boolean': (0, 1, int), 'coordinate_width': (0, self.env.grid.width, int), 'coordinate_height': (0, self.env.grid.height, int), 'agent_dir': (0, 3, int)}

        relevant_state_variables = self.state_attributes
        min_values = np.array([]); max_values = np.array([])

        for var in relevant_state_variables:
            types = state_attribute_types[var]  

            if var == 'wall_gaps':
                for _ in range(4):
                    for t in types:
                        space_param = self.gym_space_params[t]
                        min_values = np.append(min_values, space_param[0])
                        max_values = np.append(max_values, space_param[1])    
            elif var == 'walls':
                space_param = self.gym_space_params[types[0]]
                for _ in range(self.env.grid.width):
                    for _ in range(self.env.grid.height):
                        min_values = np.append(min_values, space_param[0])
                        max_values = np.append(max_values, space_param[1])
            else:
                for t in types:
                    
                    space_param = self.gym_space_params[t]

                    min_values = np.append(min_values, space_param[0])
                    max_values = np.append(max_values, space_param[1])
        
        if self.normalize_state:
            min_values = (min_values - min_values)/(max_values - min_values)
            max_values = (max_values - min_values)/(max_values - min_values)
        return gym.spaces.Box(low=min_values, high=max_values, dtype=int if not self.normalize_state else float)

    def _create_observation_space(self, state_attribute_types):
        #return the desired gym Spaces based on the observation space
        if self.observation_type == 'image':
            frame = self.env.unwrapped.get_frame(tile_size=8)
            frame = self.padding_func.pad(frame)
            return gym.spaces.Box(low=0, high=255, shape=frame.shape, dtype=np.uint8), self._create_expert_observation_space(state_attribute_types)

        elif self.observation_type == 'expert':
            expert_observation_space = self._create_expert_observation_space(state_attribute_types)
            return expert_observation_space, expert_observation_space


        elif self.observation_type == 'factored':
            raise NotImplementedError('ERROR: to be implemented after factored representation encoder')

    def get_low_high_attr(self, attr):

        limits = []

        for attr_type in self.state_attribute_types[attr]:
            attr_limit = self.gym_space_params[attr_type]
            limits.append(attr_limit)

        return limits
            
    def render(self):
        '''Implementing render() according to ABC of gymnasium env'''
        frame = self.env.unwrapped.get_frame(tile_size=8, highlight=self.highlight)
        frame = self.padding_func.pad(frame)
        frame = self.data_augmentor.apply_transformation(frame)
        return frame

    def step(self, action):
        '''
        Inputs:
        - action: integer value representing discrete actions
        0 Turn left
        1 Turn right
        2 Move forward
        3 Pick up an object
        4 Drop
        5 Toggle/activate an object
        6 Done
        '''
        (_, reward, terminated, truncated, info) = self.env.step(action)
        

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)

        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame
        info['action'] = action
        
        #apply image transformations if needed
        frame = self.data_augmentor.apply_transformation(frame)
        

        state, norm_state_array = self._construct_state()



        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the state in dictionary form for info
        info['state_dict'] = state
        #add the visual observation into the info for debugging
        info['obs'] = frame
        #store original reward in case needed
        info['original_reward'] = reward
        
        info['is_success'] = reward > 0 and terminated

        info['dist_goal'] = abs(state['agent_pos'][0]-state['goal_pos'][0]) + abs(state['agent_pos'][1]-state['goal_pos'][1])

        
        #NOTE: replace with 0-1 reward function -- avoid accumulation of discounted reward (very small when reaching the goal)
        reward = 1.0 if reward > 0 else 0.0

        


        observation = self._get_obs(image = frame, state = norm_state_array, factored = factored)

       

        # #reset in case the environment is done
        # if done:
        #     self.reset()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        '''
        Inputs: None

        Outputs:
        - observation: visual observations after resetting the environment
        - info: additional information from environment after resetting
        '''
        
        __, info = self.env.reset(seed=seed)

        #in case reset requires new state to have controlled state factors -- implement those controls
        if self.reset_type == 'custom':
            self.env = self.custom_resetter.factored_reset(self.env, self.env.unwrapped.grid.height, self.env.unwrapped.grid.width, self.controlled_factors)
        elif self.reset_type == 'random':
            self._randomize_reset()
        

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)
        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame

        #apply image transformations if needed
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected NumPy array, but got {type(frame)}")
        frame = self.data_augmentor.apply_transformation(frame)
        
        state, norm_state_array = self._construct_state()
        
        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the visual observation into the info for debugging
        info['obs'] = frame
        #add the state in dictionary form for info
        info['state_dict'] = state

       

        
        observation = self._get_obs(image = frame, state = norm_state_array, factored = factored)

        
        
        return observation, info
    

    def get_curr_obs(self):

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)
        
        state, norm_state_array = self._construct_state()
        

        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)


        obs = self._get_obs(image=frame, state= norm_state_array, factored= factored)

        return obs

    def _factorize_obs(self, observation):
        #TODO: implement inference time call to factored representation model
        return None
    
    def _construct_state(self):

        state = {}

        #extract the types of all tiles in the grid: useful for goal, key and door position
        types = np.array([x.type if x is not None else None for x in self.env.unwrapped.grid.grid])
        types_set = set(types)
        
        grid_width = self.env.unwrapped.width
        grid_height = self.env.unwrapped.height

        for attr in self.state_attributes:

            if hasattr(self.env.unwrapped, attr):
                state[attr] = getattr(self.env.unwrapped,attr)
            elif attr == 'goal_pos':
                # query = 'goal' if 'goal' in types_set else 'box'
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='goal')[0][0]].cur_pos) 
            
            #other positional attributes for key and door
            elif ('key' in types) and (attr == 'key_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='key')[0][0]].cur_pos)
                
            elif ('key' not in types) and (attr == 'key_pos'):
                state['key_pos'] = tuple(state['agent_pos'])

            elif ('door' in types) and (attr == 'door_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].cur_pos)
            
            #other positional attributes for ball (only for obstacle env)
            elif ('ball' in types) and (attr == 'ball_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='ball')[0][0]].cur_pos)
            elif ('ball' not in types) and (attr == 'ball_pos'):
                state['ball_pos'] = tuple(state['agent_pos'])

            #other attributes like opening, holding, locked etc...
            elif (attr == 'holding_key'):
                state[attr] = int(isinstance(self.env.unwrapped.carrying, Key))
            elif ('door' in types) and (attr == 'door_locked'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_locked)
            elif ('door' in types) and (attr == 'door_open'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_open)
            #other attributes like holding object (only for obstacle env)
            elif (attr == 'holding_ball'):
                state[attr] = int(isinstance(self.env.unwrapped.carrying, Ball))
            #walls attribute for shape of grid
            elif (attr == 'walls'):
                state[attr] = list(np.where(types=='wall', 1, 0))
            
            #wall gaps for the FourRooms environment
            elif attr == 'wall_gaps':
                gap_positions = []
                mid_x = grid_width // 2
                mid_y = grid_height // 2

                # Check along the vertical midline (x = mid_x)
                for y in range(0, grid_height):  # Skip corners
                    obj = self.env.unwrapped.grid.get(mid_x, y)
                    if obj is None or isinstance(obj, Goal):
                        gap_positions+=[mid_x, y]

                # Check along the horizontal midline (y = mid_y)
                for x in range(0, grid_width):  # Skip corners
                    obj = self.env.unwrapped.grid.get(x, mid_y)
                    if obj is None or isinstance(obj, Goal):
                        gap_positions+=[x, mid_y]

                state[attr] = tuple(gap_positions)

        norm_state_array = np.array([item for key in state.keys() for item in (state[key] if isinstance(state[key], Iterable) else [state[key]])])
        
        #construct normalized state array output
        if self.normalize_state:
            min_values = np.array(self.expert_observation_space.low)
            max_values = np.array(self.expert_observation_space.high)
            norm_state_array = list((norm_state_array - min_values) / (max_values - min_values))
        return state, norm_state_array
        

    def _get_obs(self, image=None, state=None, factored=None):
        
        if self.observation_type == 'image':
            return image
        
        elif self.observation_type == 'expert':
            return state
        
        elif self.observation_type == 'factored':
            return factored
        
        else:
            raise Exception('ERROR: observation type {} undefined'.format(self.observation_type))

class CartPoleDataGenerator(DataGenerator):

    def __init__(self, cfg: DictConfig):

        super(CartPoleDataGenerator, self).__init__()

        #register BlockedDoorKey to the environment list
        register(
            id="MiniGrid-BlockedDoorKey-6x6",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 6},
        )
        register(
            id="MiniGrid-BlockedDoorKey-8x8",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 8},
        )
        register(
            id="MiniGrid-BlockedDoorKey-16x16",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 16},
        )

        # store configs from terminal input
        configs = cfg
        
        # Configs from the yaml file
        self.observation_type = configs['observation_space']
        self.state_attributes = configs['state_attributes']
        self.state_attribute_types = configs['state_attribute_types']
        self.reset_type = configs['reset_type']
        self.normalize_state = configs['normalize_state']

        # Create the environment
        self.env = gym.make(configs['environment_name'], render_mode='rgb_array', max_episode_steps = configs['max_steps'], max_steps=configs['max_steps'])

        # Remove the white background from the visual environment
        self.highlight = configs['highlight']

        # Optionally wrap the environment for fully observable states
        self.env = FullyObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)

        
        # Wrap the environment to enable stochastic actions
        if configs['deterministic_action'] is False:
            self.env = StochasticActionWrapper(env=self.env, prob=configs['action_stochasticity'])
        
        # Wrap the environment to enable exploration bonus if needed
        if configs['exploration_bonus'] is True:
            self.env = ActionBonus(env = self.env)

        #Store the controlled factors array
        self.controlled_factors = configs['controlled_factors']
        
        self.render_mode = 'rgb_array'

        #storing the custom reset function if needed
        self.custom_resetter = CustomEnvReset(configs['environment_name'], configs['state_attributes'])

        # Reset the environment to initialize it
        self.env.reset()
        
        #prepare the data augmentor and padding function instance 
        self.data_augmentor = DataAugmentor(configs['transformations'])
        self.padding_func = FramePadder(configs['padded_size'])

        #creating the observation space and actions space
        self.action_space = self.env.action_space
        self.observation_space, self.expert_observation_space = self._create_observation_space(configs['state_attribute_types'])


        #creating other gym environment attributes
        self.spec = self.env.spec
        self.metadata = self.env.metadata
        self.np_random = self.env.np_random
        

    def _create_expert_observation_space(self, state_attribute_types):

        self.gym_space_params = {'boolean': (0, 1, int), 'coordinate_width': (0, self.env.grid.width, int), 'coordinate_height': (0, self.env.grid.height, int), 'agent_dir': (0, 3, int)}

        relevant_state_variables = self.state_attributes
        min_values = np.array([]); max_values = np.array([])

        for var in relevant_state_variables:
            types = state_attribute_types[var]  

            if var == 'wall_gaps':
                for _ in range(4):
                    for t in types:
                        space_param = self.gym_space_params[t]
                        min_values = np.append(min_values, space_param[0])
                        max_values = np.append(max_values, space_param[1])    
            elif var == 'walls':
                space_param = self.gym_space_params[types[0]]
                for _ in range(self.env.grid.width):
                    for _ in range(self.env.grid.height):
                        min_values = np.append(min_values, space_param[0])
                        max_values = np.append(max_values, space_param[1])
            else:
                for t in types:
                    
                    space_param = self.gym_space_params[t]

                    min_values = np.append(min_values, space_param[0])
                    max_values = np.append(max_values, space_param[1])
        
        if self.normalize_state:
            min_values = (min_values - min_values)/(max_values - min_values)
            max_values = (max_values - min_values)/(max_values - min_values)
        return gym.spaces.Box(low=min_values, high=max_values, dtype=int if not self.normalize_state else float)

    def _create_observation_space(self, state_attribute_types):
        #return the desired gym Spaces based on the observation space
        if self.observation_type == 'image':
            frame = self.env.unwrapped.get_frame(tile_size=8)
            frame = self.padding_func.pad(frame)
            return gym.spaces.Box(low=0, high=255, shape=frame.shape, dtype=np.uint8), self._create_expert_observation_space(state_attribute_types)

        elif self.observation_type == 'expert':
            expert_observation_space = self._create_expert_observation_space(state_attribute_types)
            return expert_observation_space, expert_observation_space


        elif self.observation_type == 'factored':
            raise NotImplementedError('ERROR: to be implemented after factored representation encoder')

    def get_low_high_attr(self, attr):

        limits = []

        for attr_type in self.state_attribute_types[attr]:
            attr_limit = self.gym_space_params[attr_type]
            limits.append(attr_limit)

        return limits
            
    def render(self):
        '''Implementing render() according to ABC of gymnasium env'''
        frame = self.env.unwrapped.get_frame(tile_size=8, highlight=self.highlight)
        frame = self.padding_func.pad(frame)
        frame = self.data_augmentor.apply_transformation(frame)
        return frame

    def step(self, action):
        '''
        Inputs:
        - action: integer value representing discrete actions
        0 Turn left
        1 Turn right
        2 Move forward
        3 Pick up an object
        4 Drop
        5 Toggle/activate an object
        6 Done
        '''
        (_, reward, terminated, truncated, info) = self.env.step(action)
        

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)

        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame
        info['action'] = action
        
        #apply image transformations if needed
        frame = self.data_augmentor.apply_transformation(frame)
        

        state, norm_state_array = self._construct_state()



        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the state in dictionary form for info
        info['state_dict'] = state
        #add the visual observation into the info for debugging
        info['obs'] = frame
        #store original reward in case needed
        info['original_reward'] = reward
        
        info['is_success'] = reward > 0 and terminated

        info['dist_goal'] = abs(state['agent_pos'][0]-state['goal_pos'][0]) + abs(state['agent_pos'][1]-state['goal_pos'][1])

        
        #NOTE: replace with 0-1 reward function -- avoid accumulation of discounted reward (very small when reaching the goal)
        reward = 1.0 if reward > 0 else 0.0

        


        observation = self._get_obs(image = frame, state = norm_state_array, factored = factored)

       

        # #reset in case the environment is done
        # if done:
        #     self.reset()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        '''
        Inputs: None

        Outputs:
        - observation: visual observations after resetting the environment
        - info: additional information from environment after resetting
        '''
        
        __, info = self.env.reset(seed=seed)

        #in case reset requires new state to have controlled state factors -- implement those controls
        if self.reset_type == 'custom':
            self.env = self.custom_resetter.factored_reset(self.env, self.env.unwrapped.grid.height, self.env.unwrapped.grid.width, self.controlled_factors)
        elif self.reset_type == 'random':
            self._randomize_reset()
        

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)
        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame

        #apply image transformations if needed
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected NumPy array, but got {type(frame)}")
        frame = self.data_augmentor.apply_transformation(frame)
        
        state, norm_state_array = self._construct_state()
        
        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the visual observation into the info for debugging
        info['obs'] = frame
        #add the state in dictionary form for info
        info['state_dict'] = state

       

        
        observation = self._get_obs(image = frame, state = norm_state_array, factored = factored)

        
        
        return observation, info
    

    def get_curr_obs(self):

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)
        
        state, norm_state_array = self._construct_state()
        

        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)


        obs = self._get_obs(image=frame, state= norm_state_array, factored= factored)

        return obs

    def _factorize_obs(self, observation):
        #TODO: implement inference time call to factored representation model
        return None
    
    def _construct_state(self):

        state = {}

        #extract the types of all tiles in the grid: useful for goal, key and door position
        types = np.array([x.type if x is not None else None for x in self.env.unwrapped.grid.grid])
        types_set = set(types)
        
        grid_width = self.env.unwrapped.width
        grid_height = self.env.unwrapped.height

        for attr in self.state_attributes:

            if hasattr(self.env.unwrapped, attr):
                state[attr] = getattr(self.env.unwrapped,attr)
            elif attr == 'goal_pos':
                # query = 'goal' if 'goal' in types_set else 'box'
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='goal')[0][0]].cur_pos) 
            
            #other positional attributes for key and door
            elif ('key' in types) and (attr == 'key_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='key')[0][0]].cur_pos)
                
            elif ('key' not in types) and (attr == 'key_pos'):
                state['key_pos'] = tuple(state['agent_pos'])

            elif ('door' in types) and (attr == 'door_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].cur_pos)
            
            #other positional attributes for ball (only for obstacle env)
            elif ('ball' in types) and (attr == 'ball_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='ball')[0][0]].cur_pos)
            elif ('ball' not in types) and (attr == 'ball_pos'):
                state['ball_pos'] = tuple(state['agent_pos'])

            #other attributes like opening, holding, locked etc...
            elif (attr == 'holding_key'):
                state[attr] = int(isinstance(self.env.unwrapped.carrying, Key))
            elif ('door' in types) and (attr == 'door_locked'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_locked)
            elif ('door' in types) and (attr == 'door_open'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_open)
            #other attributes like holding object (only for obstacle env)
            elif (attr == 'holding_ball'):
                state[attr] = int(isinstance(self.env.unwrapped.carrying, Ball))
            #walls attribute for shape of grid
            elif (attr == 'walls'):
                state[attr] = list(np.where(types=='wall', 1, 0))
            
            #wall gaps for the FourRooms environment
            elif attr == 'wall_gaps':
                gap_positions = []
                mid_x = grid_width // 2
                mid_y = grid_height // 2

                # Check along the vertical midline (x = mid_x)
                for y in range(0, grid_height):  # Skip corners
                    obj = self.env.unwrapped.grid.get(mid_x, y)
                    if obj is None or isinstance(obj, Goal):
                        gap_positions+=[mid_x, y]

                # Check along the horizontal midline (y = mid_y)
                for x in range(0, grid_width):  # Skip corners
                    obj = self.env.unwrapped.grid.get(x, mid_y)
                    if obj is None or isinstance(obj, Goal):
                        gap_positions+=[x, mid_y]

                state[attr] = tuple(gap_positions)

        norm_state_array = np.array([item for key in state.keys() for item in (state[key] if isinstance(state[key], Iterable) else [state[key]])])
        
        #construct normalized state array output
        if self.normalize_state:
            min_values = np.array(self.expert_observation_space.low)
            max_values = np.array(self.expert_observation_space.high)
            norm_state_array = list((norm_state_array - min_values) / (max_values - min_values))
        return state, norm_state_array
        

    def _get_obs(self, image=None, state=None, factored=None):
        
        if self.observation_type == 'image':
            return image
        
        elif self.observation_type == 'expert':
            return state
        
        elif self.observation_type == 'factored':
            return factored
        
        else:
            raise Exception('ERROR: observation type {} undefined'.format(self.observation_type))

class OGBenchDataGenerator(DataGenerator):

    def __init__(self, cfg: DictConfig):

        super(OGBenchDataGenerator, self).__init__()

        #register BlockedDoorKey to the environment list
        register(
            id="MiniGrid-BlockedDoorKey-6x6",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 6},
        )
        register(
            id="MiniGrid-BlockedDoorKey-8x8",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 8},
        )
        register(
            id="MiniGrid-BlockedDoorKey-16x16",
            entry_point="data.custom_env.blockeddoorkey:BlockedDoorKeyEnv",
            kwargs={"size": 16},
        )

        # Parse yaml file parameters for data generator
        config = cfg
        
        # Configs from the yaml file
        self.observation_type = configs['observation_space']
        self.state_attributes = configs['state_attributes']
        self.state_attribute_types = configs['state_attribute_types']
        self.reset_type = configs['reset_type']
        self.normalize_state = configs['normalize_state']

        # Create the environment
        self.env = gym.make(configs['environment_name'], render_mode='rgb_array', max_episode_steps = configs['max_steps'], max_steps=configs['max_steps'])

        # Remove the white background from the visual environment
        self.highlight = configs['highlight']

        # Optionally wrap the environment for fully observable states
        self.env = FullyObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)

        
        # Wrap the environment to enable stochastic actions
        if configs['deterministic_action'] is False:
            self.env = StochasticActionWrapper(env=self.env, prob=configs['action_stochasticity'])
        
        # Wrap the environment to enable exploration bonus if needed
        if configs['exploration_bonus'] is True:
            self.env = ActionBonus(env = self.env)

        #Store the controlled factors array
        self.controlled_factors = configs['controlled_factors']
        
        self.render_mode = 'rgb_array'

        #storing the custom reset function if needed
        self.custom_resetter = CustomEnvReset(configs['environment_name'], configs['state_attributes'])

        # Reset the environment to initialize it
        self.env.reset()
        
        #prepare the data augmentor and padding function instance 
        self.data_augmentor = DataAugmentor(configs['transformations'])
        self.padding_func = FramePadder(configs['padded_size'])

        #creating the observation space and actions space
        self.action_space = self.env.action_space
        self.observation_space, self.expert_observation_space = self._create_observation_space(configs['state_attribute_types'])


        #creating other gym environment attributes
        self.spec = self.env.spec
        self.metadata = self.env.metadata
        self.np_random = self.env.np_random
        

    def _create_expert_observation_space(self, state_attribute_types):

        self.gym_space_params = {'boolean': (0, 1, int), 'coordinate_width': (0, self.env.grid.width, int), 'coordinate_height': (0, self.env.grid.height, int), 'agent_dir': (0, 3, int)}

        relevant_state_variables = self.state_attributes
        min_values = np.array([]); max_values = np.array([])

        for var in relevant_state_variables:
            types = state_attribute_types[var]  

            if var == 'wall_gaps':
                for _ in range(4):
                    for t in types:
                        space_param = self.gym_space_params[t]
                        min_values = np.append(min_values, space_param[0])
                        max_values = np.append(max_values, space_param[1])    
            elif var == 'walls':
                space_param = self.gym_space_params[types[0]]
                for _ in range(self.env.grid.width):
                    for _ in range(self.env.grid.height):
                        min_values = np.append(min_values, space_param[0])
                        max_values = np.append(max_values, space_param[1])
            else:
                for t in types:
                    
                    space_param = self.gym_space_params[t]

                    min_values = np.append(min_values, space_param[0])
                    max_values = np.append(max_values, space_param[1])
        
        if self.normalize_state:
            min_values = (min_values - min_values)/(max_values - min_values)
            max_values = (max_values - min_values)/(max_values - min_values)
        return gym.spaces.Box(low=min_values, high=max_values, dtype=int if not self.normalize_state else float)

    def _create_observation_space(self, state_attribute_types):
        #return the desired gym Spaces based on the observation space
        if self.observation_type == 'image':
            frame = self.env.unwrapped.get_frame(tile_size=8)
            frame = self.padding_func.pad(frame)
            return gym.spaces.Box(low=0, high=255, shape=frame.shape, dtype=np.uint8), self._create_expert_observation_space(state_attribute_types)

        elif self.observation_type == 'expert':
            expert_observation_space = self._create_expert_observation_space(state_attribute_types)
            return expert_observation_space, expert_observation_space


        elif self.observation_type == 'factored':
            raise NotImplementedError('ERROR: to be implemented after factored representation encoder')

    def get_low_high_attr(self, attr):

        limits = []

        for attr_type in self.state_attribute_types[attr]:
            attr_limit = self.gym_space_params[attr_type]
            limits.append(attr_limit)

        return limits
            
    def render(self):
        '''Implementing render() according to ABC of gymnasium env'''
        frame = self.env.unwrapped.get_frame(tile_size=8, highlight=self.highlight)
        frame = self.padding_func.pad(frame)
        frame = self.data_augmentor.apply_transformation(frame)
        return frame

    def step(self, action):
        '''
        Inputs:
        - action: integer value representing discrete actions
        0 Turn left
        1 Turn right
        2 Move forward
        3 Pick up an object
        4 Drop
        5 Toggle/activate an object
        6 Done
        '''
        (_, reward, terminated, truncated, info) = self.env.step(action)
        

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)

        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame
        info['action'] = action
        
        #apply image transformations if needed
        frame = self.data_augmentor.apply_transformation(frame)
        

        state, norm_state_array = self._construct_state()



        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the state in dictionary form for info
        info['state_dict'] = state
        #add the visual observation into the info for debugging
        info['obs'] = frame
        #store original reward in case needed
        info['original_reward'] = reward
        
        info['is_success'] = reward > 0 and terminated

        info['dist_goal'] = abs(state['agent_pos'][0]-state['goal_pos'][0]) + abs(state['agent_pos'][1]-state['goal_pos'][1])

        
        #NOTE: replace with 0-1 reward function -- avoid accumulation of discounted reward (very small when reaching the goal)
        reward = 1.0 if reward > 0 else 0.0

        


        observation = self._get_obs(image = frame, state = norm_state_array, factored = factored)

       

        # #reset in case the environment is done
        # if done:
        #     self.reset()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        '''
        Inputs: None

        Outputs:
        - observation: visual observations after resetting the environment
        - info: additional information from environment after resetting
        '''
        
        __, info = self.env.reset(seed=seed)

        #in case reset requires new state to have controlled state factors -- implement those controls
        if self.reset_type == 'custom':
            self.env = self.custom_resetter.factored_reset(self.env, self.env.unwrapped.grid.height, self.env.unwrapped.grid.width, self.controlled_factors)
        elif self.reset_type == 'random':
            self._randomize_reset()
        

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)
        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame

        #apply image transformations if needed
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected NumPy array, but got {type(frame)}")
        frame = self.data_augmentor.apply_transformation(frame)
        
        state, norm_state_array = self._construct_state()
        
        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the visual observation into the info for debugging
        info['obs'] = frame
        #add the state in dictionary form for info
        info['state_dict'] = state

       

        
        observation = self._get_obs(image = frame, state = norm_state_array, factored = factored)

        
        
        return observation, info
    

    def get_curr_obs(self):

        frame = self.env.unwrapped.get_frame(tile_size=8)
        frame = self.padding_func.pad(frame)
        
        state, norm_state_array = self._construct_state()
        

        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)


        obs = self._get_obs(image=frame, state= norm_state_array, factored= factored)

        return obs

    def _factorize_obs(self, observation):
        #TODO: implement inference time call to factored representation model
        return None
    
    def _construct_state(self):

        state = {}

        #extract the types of all tiles in the grid: useful for goal, key and door position
        types = np.array([x.type if x is not None else None for x in self.env.unwrapped.grid.grid])
        types_set = set(types)
        
        grid_width = self.env.unwrapped.width
        grid_height = self.env.unwrapped.height

        for attr in self.state_attributes:

            if hasattr(self.env.unwrapped, attr):
                state[attr] = getattr(self.env.unwrapped,attr)
            elif attr == 'goal_pos':
                # query = 'goal' if 'goal' in types_set else 'box'
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='goal')[0][0]].cur_pos) 
            
            #other positional attributes for key and door
            elif ('key' in types) and (attr == 'key_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='key')[0][0]].cur_pos)
                
            elif ('key' not in types) and (attr == 'key_pos'):
                state['key_pos'] = tuple(state['agent_pos'])

            elif ('door' in types) and (attr == 'door_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].cur_pos)
            
            #other positional attributes for ball (only for obstacle env)
            elif ('ball' in types) and (attr == 'ball_pos'):
                state[attr] = tuple(self.env.unwrapped.grid.grid[np.where(types=='ball')[0][0]].cur_pos)
            elif ('ball' not in types) and (attr == 'ball_pos'):
                state['ball_pos'] = tuple(state['agent_pos'])

            #other attributes like opening, holding, locked etc...
            elif (attr == 'holding_key'):
                state[attr] = int(isinstance(self.env.unwrapped.carrying, Key))
            elif ('door' in types) and (attr == 'door_locked'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_locked)
            elif ('door' in types) and (attr == 'door_open'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_open)
            #other attributes like holding object (only for obstacle env)
            elif (attr == 'holding_ball'):
                state[attr] = int(isinstance(self.env.unwrapped.carrying, Ball))
            #walls attribute for shape of grid
            elif (attr == 'walls'):
                state[attr] = list(np.where(types=='wall', 1, 0))
            
            #wall gaps for the FourRooms environment
            elif attr == 'wall_gaps':
                gap_positions = []
                mid_x = grid_width // 2
                mid_y = grid_height // 2

                # Check along the vertical midline (x = mid_x)
                for y in range(0, grid_height):  # Skip corners
                    obj = self.env.unwrapped.grid.get(mid_x, y)
                    if obj is None or isinstance(obj, Goal):
                        gap_positions+=[mid_x, y]

                # Check along the horizontal midline (y = mid_y)
                for x in range(0, grid_width):  # Skip corners
                    obj = self.env.unwrapped.grid.get(x, mid_y)
                    if obj is None or isinstance(obj, Goal):
                        gap_positions+=[x, mid_y]

                state[attr] = tuple(gap_positions)

        norm_state_array = np.array([item for key in state.keys() for item in (state[key] if isinstance(state[key], Iterable) else [state[key]])])
        
        #construct normalized state array output
        if self.normalize_state:
            min_values = np.array(self.expert_observation_space.low)
            max_values = np.array(self.expert_observation_space.high)
            norm_state_array = list((norm_state_array - min_values) / (max_values - min_values))
        return state, norm_state_array
        

    def _get_obs(self, image=None, state=None, factored=None):
        
        if self.observation_type == 'image':
            return image
        
        elif self.observation_type == 'expert':
            return state
        
        elif self.observation_type == 'factored':
            return factored
        
        else:
            raise Exception('ERROR: observation type {} undefined'.format(self.observation_type))

def build_data_generator(config_filename:str):

    if 'doorkey' in config_filename or 'empty' in config_filename
    data_generator = MiniGridDataGenerator(config_filename=config_filename)

if __name__ == '__main__':

    pdb.set_trace()

    data_generator = DataGenerator()
    data_gen_test = DataGenerator(config_filename='config_test.yaml')
    
    obs, info = data_generator.reset()
    obs_t, info_t = data_gen_test.reset()


    pdb.set_trace()

    max_steps = 1000
    step = 0

    base_path = os.path.join('./temp/')
    os.makedirs(base_path, exist_ok=True)

    while step < max_steps:

        # act = int(input('Action: '))
        act = data_generator.action_space.sample()

        obs, rew, term, trunc, info = data_generator.step(act)
        
        print('REWARD: ', rew)
        print('TERM: ', term)
        if trunc:
            print("TRUNCATED EARLY")
            pdb.set_trace()
        img = Image.fromarray(info['obs'])
        img.save(os.path.join(base_path, f'step_{step}.png'))

        step += 1
    
    pdb.set_trace()

    MAX_STEPS = 5

    temp_dir = os.path.relpath('./temp_obs')
    os.makedirs(temp_dir, exist_ok=True)

    for j in range(3):
        # pdb.set_trace()
        obs, info = data_generator.reset(seed=j)
        img = Image.fromarray(info['obs'])
        img.save(os.path.join(temp_dir, 'reset_test.jpeg'))

        for i in range(MAX_STEPS):

            rand_action = 6

            while (rand_action == 6):
            
                rand_action = data_generator.env.action_space.sample()

            observation, reward, terminated, truncated, info = data_generator.step(rand_action)

            print('Current State :', observation)
            print('Info: ', info['state_dict'])
            print('Reward: ', reward)

            img = Image.fromarray(info['obs'])

            img.save(os.path.join(temp_dir, '{}_modified.jpeg'.format(i)))

            # img = Image.fromarray(info['original_obs'])

            # img.save(os.path.join(temp_dir, '{}_original.jpeg'.format(i)))
        
        

